import cupy as cp
import cupyx.scipy.ndimage as ndi
import imageio.v3 as iio
import argparse
import time
import csv
import os

# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------

def psnr(img1, img2):
    mse = cp.mean((img1 - img2) ** 2)
    if float(mse) == 0.0:
        return cp.inf
    return -10.0 * cp.log10(mse)

def ssim_gray(img1, img2):
    K1, K2 = 0.01, 0.03
    L = 1.0
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    mu1 = ndi.gaussian_filter(img1, sigma=1.5, mode="nearest")
    mu2 = ndi.gaussian_filter(img2, sigma=1.5, mode="nearest")

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = ndi.gaussian_filter(img1 * img1, sigma=1.5, mode="nearest") - mu1_sq
    sigma2_sq = ndi.gaussian_filter(img2 * img2, sigma=1.5, mode="nearest") - mu2_sq
    sigma12 = ndi.gaussian_filter(img1 * img2, sigma=1.5, mode="nearest") - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return cp.mean(ssim_map)

def ssim(img1, img2):
    if img1.ndim == 2:
        return ssim_gray(img1, img2)
    total = 0
    for c in range(img1.shape[2]):
        total += ssim_gray(img1[:, :, c], img2[:, :, c])
    return total / img1.shape[2]

# ---------------------------------------------------------
# Utility
# ---------------------------------------------------------

def load_image(path):
    import imageio.v3 as iio
    import cupy as cp

    img_cpu = iio.imread(path).astype("float32")

    # Normalizza se necessario
    if img_cpu.max() > 1.0:
        img_cpu /= 255.0

    # Porta su GPU
    img = cp.asarray(img_cpu)

    # Converti in grayscale su GPU
    if img.ndim == 3:
        img = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]

    return img


def save_image(path, img):
    iio.imwrite(path, (cp.asnumpy(img) * 255).astype("uint8"))

def timer(func, *args, **kwargs):
    cp.cuda.Stream.null.synchronize()
    t0 = time.time()
    out = func(*args, **kwargs)
    cp.cuda.Stream.null.synchronize()
    return out, time.time() - t0

# ---------------------------------------------------------
# 1) Gaussian
# ---------------------------------------------------------
def gaussian_denoise(img, sigma=2.0):
    return ndi.gaussian_filter(img, sigma=sigma, mode="nearest")

# ---------------------------------------------------------
# 2) Median
# ---------------------------------------------------------
def median_denoise(img, size=3):
    return ndi.median_filter(img, size=size, mode="nearest")

# ---------------------------------------------------------
# 3) Mean / Box blur
# ---------------------------------------------------------
def mean_denoise(img, size=5):
    kernel = cp.ones((size, size), dtype=cp.float32) / (size * size)

    if img.ndim == 2:  # grayscale
        return ndi.convolve(img, kernel, mode="nearest")

    # RGB: canale per canale
    out = cp.zeros_like(img)
    for c in range(img.shape[2]):
        out[:, :, c] = ndi.convolve(img[:, :, c], kernel, mode="nearest")
    return out

# ---------------------------------------------------------
# 4) FFT low-pass denoising
# ---------------------------------------------------------
def save_fft_image(path, fft_data):
    # Magnitudine logaritmica per visualizzare bene
    mag = cp.log1p(cp.abs(fft_data))
    mag_cpu = cp.asnumpy(mag)
    mag_cpu = (mag_cpu / mag_cpu.max() * 255).astype("uint8")
    iio.imwrite(path, mag_cpu)


def _fft_denoise_gray(channel, cutoff, base_name):
    # FFT 2D
    f = cp.fft.fft2(channel)
    fshift = cp.fft.fftshift(f)

    # Salva FFT originale
    save_fft_image(f"{base_name}_fft_original.png", fshift)

    h, w = channel.shape
    cy, cx = h // 2, w // 2

    # Maschera passa-basso
    mask = cp.zeros((h, w), dtype=cp.float32)
    radius = int(min(h, w) * cutoff)

    Y, X = cp.ogrid[:h, :w]
    dist = cp.sqrt((X - cx)**2 + (Y - cy)**2)
    mask[dist <= radius] = 1.0

    # FFT filtrata
    fshift_filtered = fshift * mask

    # Salva FFT filtrata
    save_fft_image(f"{base_name}_fft_filtered.png", fshift_filtered)

    # Inversa
    f_ishift = cp.fft.ifftshift(fshift_filtered)
    img_back = cp.fft.ifft2(f_ishift).real

    return img_back


def fft_denoise(img, cutoff, base_name):
    if img.ndim == 2:
        return _fft_denoise_gray(img, cutoff, base_name)

    # RGB: canale per canale
    out = cp.zeros_like(img)
    for c in range(img.shape[2]):
        out[:, :, c] = _fft_denoise_gray(img[:, :, c], cutoff, base_name + f"_c{c}")
    return out

# ---------------------------------------------------------
# 5) Bilateral 
# ---------------------------------------------------------

def bilateral_denoise(img, sigma_s=3.0, sigma_r=0.1):
    h, w = img.shape
    radius = int(3 * sigma_s)

    # Coordinate della finestra
    Y, X = cp.mgrid[-radius:radius+1, -radius:radius+1]
    spatial = cp.exp(-(X**2 + Y**2) / (2 * sigma_s**2))

    out = cp.zeros_like(img)

    for i in range(h):
        for j in range(w):
            y1 = max(0, i - radius)
            y2 = min(h, i + radius + 1)
            x1 = max(0, j - radius)
            x2 = min(w, j + radius + 1)

            region = img[y1:y2, x1:x2]

            # Taglia la finestra spaziale
            sy1 = y1 - (i - radius)
            sy2 = sy1 + region.shape[0]
            sx1 = x1 - (j - radius)
            sx2 = sx1 + region.shape[1]

            spatial_win = spatial[sy1:sy2, sx1:sx2]

            # Range kernel
            range_win = cp.exp(-(region - img[i, j])**2 / (2 * sigma_r**2))

            # Bilateral
            weights = spatial_win * range_win
            out[i, j] = cp.sum(weights * region) / cp.sum(weights)

    return out


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", help="Path dell'immagine da denoising")
    args = parser.parse_args()

    img_path = args.image_path
    img = load_image(img_path)

    print("GPU:", cp.cuda.runtime.getDeviceProperties(0)['name'])
    print("Immagine:", img.shape)

    base = os.path.splitext(os.path.basename(img_path))[0]

    timings = []

    # Gaussian
    out, t = timer(gaussian_denoise, img, sigma=2.0)
    save_image(f"{base}_gaussian.png", out)
    psnr_value = float(psnr(img, out))
    ssim_value = float(ssim(img, out))

    timings.append(["gaussian", t, psnr_value, ssim_value])
    
    # Median
    out, t = timer(median_denoise, img, size=3)
    save_image(f"{base}_median.png", out)
    psnr_value = float(psnr(img, out))
    ssim_value = float(ssim(img, out))

    timings.append(["median", t, psnr_value, ssim_value])

    
    # Mean
    out, t = timer(mean_denoise, img, size=5)
    save_image(f"{base}_mean.png", out)
    
    psnr_value = float(psnr(img, out))
    ssim_value = float(ssim(img, out))

    timings.append(["mean", t, psnr_value, ssim_value])
    
    # FFT
    out_fft, t = timer(fft_denoise, img, cutoff=0.1, base_name=base)

    save_image(f"{base}_fft.png", out_fft)
    psnr_value = float(psnr(img, out_fft))
    ssim_value = float(ssim(img, out_fft))

    timings.append(["fft", t, psnr_value, ssim_value])
    
    # Bilateral
    
    # CuPy non ha una funzione per il bilateral filter
    
    # out, t = timer(bilateral_denoise, img, sigma_s=3.0, sigma_r=0.1)
    # save_image(f"{base}_bilateral.png", out)

    # psnr_value = float(psnr(img, out))
    # ssim_value = float(ssim(img, out))

    # timings.append(["bilateral", t, psnr_value, ssim_value])

    
    # Salva CSV
    with open("timings.csv", "w", newline="") as f:
        writer = csv.writer(f,delimiter=';')
        writer.writerow(["algorithm", "time_seconds","psnr","ssim"])
        writer.writerows(timings)

    print("Denoising completato. Risultati salvati.")
    print("CSV generato: timings.csv")

if __name__ == "__main__":
    main()
