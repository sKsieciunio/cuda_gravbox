# Symulacja Grawitacji i Kolizji Cząsteczek w CUDA

## Budowanie (CMake)

Wymagania:

- NVIDIA CUDA Toolkit (wspierana wersja zgodna z Twoją kartą)
- Visual Studio 2022 (C++ Desktop) lub Build Tools
- Sterownik GPU z obsługą CUDA

Kroki (PowerShell):

```
# Configure
cmake -G "Visual Studio 17 2022" -A x64 -B build/vs/vs2022-release -DCMAKE_CUDA_ARCHITECTURES=75

# Build
cmake --build build/vs/vs2022-release --config Release

# Run
./build/vs/vs2022-release/Release/cuda_gravbox.exe
```

Uwaga: jeśli masz wiele wersji CUDA lub niestandardową instalację, ustaw zmienną środowiskową `CUDA_PATH` lub wskaż toolchain poprzez `-DCUDAToolkit_ROOT=...`.

Jeśli chcesz zmienić architekturę GPU, edytuj `CMAKE_CUDA_ARCHITECTURES` w odpowiednim preset (`native` można zastąpić np. `75;86`).

## Wprowadzenie

Projekt implementuje symulację grawitacji i kolizji między cząsteczkami w czasie rzeczywistym przy użyciu CUDA. Symulacja wykorzystuje integrację Verleta oraz przestrzenne hashowanie (spatial grid) do efektywnej detekcji kolizji.

## Architektura Rozwiązania

### Struktura Danych

- **ParticlesSoA**: Structure of Arrays przechowująca pozycje (x, y), poprzednie pozycje, prędkości i promienie cząstek
- **SimulationParams**: Parametry fizyczne (grawitacja, krok czasowy dt, tłumienie, współczynnik restytucji)
- **GridParams**: Parametry siatki przestrzennej (szerokość, wysokość, rozmiar komórki)

### Algorytm Główny

#### 1. Integracja Verleta (kernel.cu)

```
new_pos = 2 * pos - prev_pos + gravity * dt²
velocity = (new_pos - prev_pos) / (2 * dt)
```

**Zalety**:

- Stabilność numeryczna
- Naturalna konserwacja energii
- Brak potrzeby jawnego przechowywania prędkości (choć jest używana do wizualizacji)

#### 2. Spatial Grid Hashing

Cząsteczki są przypisywane do komórek siatki o rozmiarze `2 × promień_cząstki`. Proces składa się z:

**a) Przypisanie do komórek** (`assignParticlesToGridKernel`):

```
cellX = floor(pos_x / cell_size)
cellY = floor(pos_y / cell_size)
cellIndex = cellY * grid_width + cellX
```

**b) Sortowanie** (Thrust):

- Sortowanie tablicy `particleGridIndex` wraz z indeksami cząstek
- Grupowanie cząstek w tej samej komórce razem

**c) Wyznaczanie granic komórek** (`findCellBoundsKernel`):

- Znajdowanie początku (`gridCellStart`) i końca (`gridCellEnd`) każdej komórki w posortowanej tablicy

#### 3. Detekcja i Rozwiązywanie Kolizji (`handleCollisionsKernel`)

Dla każdej cząstki:

1. Sprawdzenie 9 sąsiednich komórek (3×3 neighborhood)
2. Test kolizji tylko z cząstkami w tych komórkach
3. Jeśli `dist < r₁ + r₂`:
   ```
   overlap = (r₁ + r₂) - dist
   separation = normal × overlap × 0.5
   ```
4. Użycie `atomicAdd` do rozdzielenia cząstek (każda przesuwa się o połowę nakładania)

**Optymalizacja**: `if (particleIdx >= otherParticleIdx) continue` - unika podwójnego sprawdzania par

#### 4. Kolizje ze Ścianami

Odbicia od granic symulacji z tłumieniem energii:

```
if (pos - radius < 0):
    pos = radius
    prev_pos = pos + velocity × dampening
```

### Pętla Symulacyjna

```
for iter in range(3):
    1. Przypisz cząstki do siatki
    2. Sortuj według indeksów komórek
    3. Wyznacz granice komórek
    4. Rozwiąż kolizje
    5. Aktualizuj pozycje (Verlet)
```

**Wielokrotne iteracje** (3×) poprawiają dokładność rozwiązywania kolizji w przypadku nakładania się wielu cząstek.

## Implementacja CUDA

### Równoległość

- Jeden wątek na cząstkę
- Rozmiar bloku: 256 wątków
- Liczba bloków: `⌈numParticles / 256⌉`

### Interoperacyjność CUDA-OpenGL

- `cudaGraphicsGLRegisterBuffer`: Rejestracja VBO do bezpośredniego dostępu przez CUDA
- `cudaGraphicsMapResources`: Mapowanie przed modyfikacją
- Brak kopiowania CPU↔GPU dla pozycji i prędkości

### Użyte Biblioteki

- **Thrust**: Sortowanie na GPU (`thrust::sort_by_key`)
- **GLEW/GLFW**: Rendering OpenGL
- **ImGui**: Interfejs użytkownika

## Złożoność Obliczeniowa

- **Naiwna detekcja**: O(n²)
- **Z spatial grid**: O(n × k), gdzie k = średnia liczba cząstek w 9 sąsiednich komórkach
- **W praktyce**: ~O(n) dla równomiernego rozkładu
