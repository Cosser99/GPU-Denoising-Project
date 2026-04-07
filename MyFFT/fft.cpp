//  Da fare:
//  -Inserire Padding
//  -Ottimizzare O(Nlog(n)) con algoritmo Cooley–Tukey
//  -Controllare la dimensione di output (forse deve essere più grande)
//  -Inserire il filtro o lasciarlo fare da un programma esterno
//  
//
//

#include <opencv2/opencv.hpp>
#include <complex>
#include <vector>
#include <chrono>

using cd = std::complex<double>;
const double PI = acos(-1);


void fft(std::vector<cd> & a, bool invert) {
    int n = a.size();

    // Bit-reversal permutation
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1)
            j ^= bit;
        j |= bit;

        if (i < j)
            std::swap(a[i], a[j]);
    }

    // FFT
    for (int len = 2; len <= n; len <<= 1) {
        double ang = 2 * PI / len * (invert ? -1 : 1);
        cd wlen(cos(ang), sin(ang));

        for (int i = 0; i < n; i += len) {
            cd w(1);
            for (int j = 0; j < len / 2; j++) {
                cd u = a[i + j];
                cd v = a[i + j + len/2] * w;

                a[i + j] = u + v;
                a[i + j + len/2] = u - v;

                w *= wlen;
            }
        }
    }

    // Normalizzazione per IFFT
    if (invert) {
        for (cd & x : a)
            x /= n;
    }
}

void fft2D(std::vector<std::vector<cd>> &img, bool invert) {
    int h = img.size();
    int w = img[0].size();

    // FFT su ogni riga
    for (int i = 0; i < h; i++)
        fft(img[i], invert);

    // FFT su ogni colonna
    for (int j = 0; j < w; j++) {
        std::vector<cd> col(h);
        for (int i = 0; i < h; i++)
            col[i] = img[i][j];

        fft(col, invert);

        for (int i = 0; i < h; i++)
            img[i][j] = col[i];
    }
}
void fftShift(cv::Mat& mag) {
    int cx = mag.cols / 2;
    int cy = mag.rows / 2;

    cv::Mat q0(mag, cv::Rect(0, 0, cx, cy));
    cv::Mat q1(mag, cv::Rect(cx, 0, cx, cy));
    cv::Mat q2(mag, cv::Rect(0, cy, cx, cy));
    cv::Mat q3(mag, cv::Rect(cx, cy, cx, cy));

    cv::Mat tmp;
    q0.copyTo(tmp); q3.copyTo(q0); tmp.copyTo(q3);
    q1.copyTo(tmp); q2.copyTo(q1); tmp.copyTo(q2);
}

bool isPowerOfTwo(int n) {
    return (n & (n - 1)) == 0;
}
int main(int argc,char **argv)
{
    auto start = std::chrono::high_resolution_clock::now();

    if(argc<2)
    {
        printf("ARGUMENT: <image>");
        return 1;
    }
    cv::Mat img = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);

    if (img.empty()) {
        std::cout << "Errore caricamento immagine\n";
        return -1;
    }
    
    //Converti in numeri complessi
    int h = img.rows;
    int w = img.cols;
    //Controllo potenza di due

    if (!isPowerOfTwo(w) || !isPowerOfTwo(h)) {
        std::cout << "Dimensioni non valide per FFT\n";
        return -1;
    }


    std::vector<std::vector<std::complex<double>>> data(h, std::vector<std::complex<double>>(w));

    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            data[i][j] = std::complex<double>(img.at<uchar>(i, j), 0);
        }
    }

    //Uso funzioni

    fft2D(data, false); // FFT
printf("check ");
    //cerco di visualizzare il magnitudo in scala logaritmica
    /************************************************************* */
    cv::Mat spectrum(h, w, CV_64F);

    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            double re = data[i][j].real();
            double im = data[i][j].imag();

            double mag = sqrt(re*re + im*im);
            spectrum.at<double>(i, j) = mag;
        }
    }
    
    spectrum += 1; // evita log(0)
    cv::log(spectrum, spectrum);
    fftShift(spectrum);
    cv::normalize(spectrum, spectrum, 0, 255, cv::NORM_MINMAX);
    spectrum.convertTo(spectrum, CV_8U);
    //cv::imshow("FFT Spectrum", spectrum);
    //cv::waitKey(0);

    // oppure
    cv::imwrite("fft.png", spectrum);


    /************************************************************ */
       
        auto startfft = std::chrono::high_resolution_clock::now();
    fft2D(data, true);  // IFFT
        auto endfft = std::chrono::high_resolution_clock::now();
        std::cout << "FFT: "
          << std::chrono::duration<double, std::milli>(endfft - startfft).count()
          << " ms\n";

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Tempo: " << elapsed.count() << " secondi\n";

    //Conversione inversa

    cv::Mat output(h, w, CV_8UC1);

    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            double val = data[i][j].real();

            val = std::max(0.0, std::min(255.0, val));
            output.at<uchar>(i, j) = static_cast<unsigned char>(val);
        }
    }
    //Salva risultato
    cv::imwrite("output.png", output);

    //Calcolo errore
    //Forse ci sono errori per i floating point

    double error = 0;

    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            double diff = img.at<uchar>(i,j) - output.at<uchar>(i,j);
            error += std::abs(diff);
        }
    }

    std::cout << "Errore totale: " << error << std::endl;

}