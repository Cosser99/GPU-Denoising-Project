nvcc %1 -o %2 ^
-I"C:\opencv\build\include" ^
-I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\include" ^
-L"C:\opencv\build\x64\vc16\lib" ^
-L"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\lib\x64" ^
-lopencv_world4120 ^
-lcufft