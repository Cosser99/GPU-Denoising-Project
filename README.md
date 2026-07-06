# GPU Denoising

Suite di benchmark per filtri di denoising CPU e CUDA. Gli header in `include/filters/`
sono l'unica API pubblica: un filtro CUDA implementerà la stessa classe e gli stessi parametri
del corrispondente filtro CPU. Non esistono namespace o header CUDA duplicati.

Filtri e immagini sono generici sul tipo di pixel: `Image<T>` e `MeanFilter<T>` (come gli altri
filtri) sono attualmente istanziati per `uint8_t` e `float`. `ImageIO` resta intenzionalmente
specializzato su `ByteImage`, perché i formati immagine caricati e salvati sono a 8 bit.

## Struttura dei backend

- `src/cpu/`: baseline CPU;
- `src/cuda/`: CUDA reference, con un file `.cu` per filtro;
- `src/cuda_optimized/`: CUDA ottimizzato, con un file `.cu` per filtro.

Ogni backend definisce gli stessi simboli C++ (`idl::MeanFilter`, `idl::GaussianFilter`, e così
via), quindi le librerie dei backend sono mutuamente esclusive nello stesso eseguibile. Il
benchmark confronterà eseguibili separati, che produrranno risultati CSV nello stesso formato.

## CMake

```bash
cmake -S . -B build
cmake --build build
```

Target già disponibili:

- `idl_core`: immagine, errori e I/O comuni;
- `idl_filters_cpu`: tutte le implementazioni CPU;
- `gpu_denoising`: demo collegata alla baseline CPU.

Il test CPU del filtro media è sempre disponibile tramite `ctest --test-dir build`.

I backend completi `idl_filters_cuda_reference` e `idl_filters_cuda_optimized` implementano
tutti i filtri del backend CPU senza introdurre nuovi header. Se `BUILD_TESTING=ON`, CTest esegue
una piccola verifica numerica del filtro media per entrambi; in assenza di GPU NVIDIA il test viene
segnato come *skipped*.
