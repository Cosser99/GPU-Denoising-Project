nvcc -arch=sm_86 %1 -o %2 ^
   -I"C:\opencv\build\include" ^
   -L"C:\opencv\build\x64\vc16\lib" ^
   -l opencv_world4120
