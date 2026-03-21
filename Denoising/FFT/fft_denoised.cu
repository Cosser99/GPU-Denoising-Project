#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
using namespace cv;
using namespace std;
static void help(char ** argv)
{
    cout << endl
        <<  "This program demonstrated the use of the discrete Fourier transform (DFT). " << endl
        <<  "The dft of an image is taken and it's power spectrum is displayed."  << endl << endl
        <<  "Usage:"                                                                      << endl
        << argv[0] << " [image_name -- default lena.jpg]" << endl << endl;
}


//
//
//          FILE GIUSTO
//
//
int main(int argc, char ** argv)
{

     if(argc<3) 
    {
        printf("ERRORE : argument must be <input image.png> <radius>");
        return;
    }
    char path[64];
    snprintf(path,sizeof(path),"..\\..\\image\\%s",argv[1]);
    // carica immagine
    cv::Mat I = cv::imread(path,cv::IMREAD_GRAYSCALE);
    
    if(I.empty()) { 
        std::cerr << "Errore: immagine non trovata!" << std::endl;
        return -1;
    }

    Mat padded;                            //expand input image to optimal size
    int m = getOptimalDFTSize( I.rows );
    int n = getOptimalDFTSize( I.cols ); // on the border add zero values
    copyMakeBorder(I, padded, 0, m - I.rows, 0, n - I.cols, BORDER_CONSTANT, Scalar::all(0));
    Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
    Mat complexI;
    merge(planes, 2, complexI);         // Add to the expanded another plane with zeros
    dft(complexI, complexI);            // this way the result may fit in the source matrix


    /***********************APPLICO IL DENOISING ***********[CHAT GPT]******************************************** */ 
        // Dimensioni
    int cx1 = complexI.cols/2;
    int cy1 = complexI.rows/2;
    int D0 = atoi(argv[2]); // raggio del filtro

    // Creiamo la maschera a due canali (reale+immaginaria)
    Mat mask = Mat::zeros(complexI.size(), CV_32FC2);

    // Loop per creare il cerchio al centro
    for(int i=0; i<mask.rows; i++){
        for(int j=0; j<mask.cols; j++){
            int u = j - cx1;
            int v = i - cy1;
            if(sqrt(float(u*u + v*v)) >= D0){       //>= Filtro passa alto (In teoria)  <= Filtro passa basso (In teoria)
                mask.at<Vec2f>(i,j)[0] = 1.0; // canale reale  originale: 1.0
                mask.at<Vec2f>(i,j)[1] = 1.0; // canale immaginario originale 1.0
            }
        }
    }
    complexI = complexI.mul(mask);
        Mat invDFT;
    idft(complexI, invDFT, DFT_SCALE | DFT_REAL_OUTPUT); // otteniamo solo la parte reale
    normalize(invDFT, invDFT, 0, 1, NORM_MINMAX);       // normalizziamo per visualizzare
    imshow("Filtered Image", invDFT);

     /**************************************************************************************** */ 


    // compute the magnitude and switch to logarithmic scale
    // => log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
    split(complexI, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
    magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
    Mat magI = planes[0];
    magI += Scalar::all(1);                    // switch to logarithmic scale
    log(magI, magI);
    // crop the spectrum, if it has an odd number of rows or columns
    magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));
    // rearrange the quadrants of Fourier image  so that the origin is at the image center
    int cx = magI.cols/2;
    int cy = magI.rows/2;
    Mat q0(magI, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
    Mat q1(magI, Rect(cx, 0, cx, cy));  // Top-Right
    Mat q2(magI, Rect(0, cy, cx, cy));  // Bottom-Left
    Mat q3(magI, Rect(cx, cy, cx, cy)); // Bottom-Right
    Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);
    q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
    q2.copyTo(q1);
    tmp.copyTo(q2);
    normalize(magI, magI, 0, 1, NORM_MINMAX); // Transform the matrix with float values into a
                                            // viewable image form (float between values 0 and 1).
    imshow("Input Image"       , I   );    // Show the result
    imshow("spectrum magnitude", magI);
    waitKey();
    return EXIT_SUCCESS;
}