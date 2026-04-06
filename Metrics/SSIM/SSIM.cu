#include <opencv2/opencv.hpp>
#include <iostream>

double getSSIM(const cv::Mat& i1, const cv::Mat& i2)
{
    const double C1 = 6.5025, C2 = 58.5225;

    cv::Mat I1, I2;
    i1.convertTo(I1, CV_32F);
    i2.convertTo(I2, CV_32F);

    cv::Mat I2_2 = I2.mul(I2);
    cv::Mat I1_2 = I1.mul(I1);
    cv::Mat I1_I2 = I1.mul(I2);

    cv::Mat mu1, mu2;
    cv::GaussianBlur(I1, mu1, cv::Size(11, 11), 1.5);
    cv::GaussianBlur(I2, mu2, cv::Size(11, 11), 1.5);

    cv::Mat mu1_2 = mu1.mul(mu1);
    cv::Mat mu2_2 = mu2.mul(mu2);
    cv::Mat mu1_mu2 = mu1.mul(mu2);

    cv::Mat sigma1_2, sigma2_2, sigma12;

    cv::GaussianBlur(I1_2, sigma1_2, cv::Size(11, 11), 1.5);
    sigma1_2 -= mu1_2;

    cv::GaussianBlur(I2_2, sigma2_2, cv::Size(11, 11), 1.5);
    sigma2_2 -= mu2_2;

    cv::GaussianBlur(I1_I2, sigma12, cv::Size(11, 11), 1.5);
    sigma12 -= mu1_mu2;

    cv::Mat t1, t2, t3;

    t1 = 2 * mu1_mu2 + C1;
    t2 = 2 * sigma12 + C2;
    t3 = t1.mul(t2);

    t1 = mu1_2 + mu2_2 + C1;
    t2 = sigma1_2 + sigma2_2 + C2;
    t1 = t1.mul(t2);

    cv::Mat ssim_map;
    divide(t3, t1, ssim_map);

    cv::Scalar mssim = mean(ssim_map);

    return mssim[0]; // grayscale
}



int main(int argc,char **argv)
{
    if(argc<3)
    {
        printf("USAGE: <original> <denoised>");
        return 1;
    }
    char path1[64];
    char path2[64];
    snprintf(path1,sizeof(path1),"..\\..\\Image\\%s",argv[1]);
    snprintf(path2,sizeof(path2),"..\\..\\Image\\%s",argv[2]);
    // 1. Carica le immagini
    cv::Mat img1 = cv::imread(path1, cv::IMREAD_GRAYSCALE);   //clean.png
    cv::Mat img2 = cv::imread(path2, cv::IMREAD_GRAYSCALE);    //denoised.png

    // 2. Controllo
    if (img1.empty() || img2.empty()) {
        std::cout << "Errore nel caricamento immagini\n";
        return -1;
    }

    // 3. Devono avere stessa dimensione
    if (img1.size() != img2.size()) {
        std::cout << "Dimensioni diverse!\n";
        return -1;
    }

    // 4. Calcola SSIM
    double ssim = getSSIM(img1, img2);

    // 5. Stampa risultato
    std::cout << "SSIM: " << ssim << std::endl;

    return 0;
}