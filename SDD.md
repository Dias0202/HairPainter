# Software Design Document (SDD)
## Hair Painter — Sistema Automatizado de Segmentação e Medição de Fibrilas de Mimivirus

**Versão**: 2.6  
**Data**: 2026-06-17  
**Autor**: Gabriel Dias  
**Status**: Em desenvolvimento ativo — **U-Net (DL) rompe o teto clássico: F1@5px 0.41 vs 0.31 (Seção 11.5)**, integrada como caminho opt-in (`use_unet`). Harness experimental confirmou que pré-processamento clássico é marginal e o teto de F1≈0.30 é físico (Seção 11). Pré-processamento atualizado para CLAHE 4.0/16; parâmetros de zona/capsídeo expostos. Comprimento real do GT medido = ~80nm (corrige a suposição anterior de 140nm).

---

## 1. Visão Geral e Problema

Pesquisadores de virologia de vírus gigantes (Mimivirus) analisam imagens de Microscopia Eletrônica de Transmissão (MET). Cada partícula viral possui centenas de **fibrilas** — estruturas filamentosas que irradiam do capsídeo. O fluxo atual exige que o pesquisador:

1. Abra a imagem no PowerPoint
2. Desenhe manualmente por cima de cada fibrilas (~400–800 por imagem)
3. Estime tamanhos visualmente

**Hair Painter** automatiza esse processo. Dada uma imagem MET, o sistema:
- Detecta automaticamente estruturas semelhantes a fibrilas no anel periCapsídico
- Colore-as com opacidade variável (navy #1c3052) indicando sobreposição
- Estima comprimento real em nm de cada estrutura usando a barra de escala
- Gera 3 entregáveis padronizados por imagem + relatório JSON

---

## 2. Contexto Técnico das Imagens

| Parâmetro | Valor |
|-----------|-------|
| Microscópio | Tecnai 120 kV BioTwin |
| Magnificação | 98.000× |
| Dimensões | 1376 × 1070 px, 8-bit paletted (indexed color) |
| Formato | TIFF com stack de 3 frames (usar frame de maior variância) |
| Escala | ~1.36 px/nm (barra de escala visual "100 nm" = 136px) |
| Metadata | Tag TIFF 65450/65451 contém pixel size físico |
| Barra de escala | Região escura nas últimas ~30 linhas (fundo preto, barra branca, texto branco) |

**SVGs de ground truth** (`Data/Manual_paint/`):
- Viewport: 1280 × 720 px → **mapeamento 1:1 para o espaço de imagem** (sem scaling)
- ~392–790 fibrilas por imagem como `<path>` bezier cúbico
- Stroke: `#1c3052` (navy), stroke-width: 1.33px
- Os SVGs anotam apenas os primeiros 1280 × 720 px da imagem (top-left)

**Anatomia do Mimivirus no MET**:
- Capsídeo: região circular densa (escura em bright-field TEM), raio ≈ 190-225 px na imagem
- Fibrilas: proteínas radiais que irradiam do capsídeo, ligeiramente mais escuras que o fundo
- Contraste individual fibrilas/fundo: **~17 unidades** (0-255) com desvio padrão ≈ 38
- Separação inter-fibrilas na superfície do capsídeo: **~3 px** = 2.2 nm (abaixo da resolução prática do filtro)

---

## 3. Requisitos Funcionais

| ID | Requisito | Estado |
|----|-----------|--------|
| RF-01 | Aceitar imagens .tif, .jpg, .png | ✅ Implementado |
| RF-02 | Processar em modo batch | ✅ Implementado |
| RF-03 | Detectar barra de escala via EasyOCR | ✅ Implementado |
| RF-04 | Fallback de escala via tag TIFF | ✅ Implementado |
| RF-05 | Detectar e mascarar capsídeo | ✅ Implementado |
| RF-06 | Segmentar estruturas de fibrilas por instância | ✅ Implementado (v2.1: ancoragem + polar Frangi) |
| RF-07 | Medir comprimento real em nm | ✅ Implementado |
| RF-08 | Entregável 1: fundo preto + fibrilas navy | ✅ Implementado |
| RF-09 | Entregável 2: raw + overlay | ✅ Implementado |
| RF-10 | Entregável 3: overlay + anotações mín/média/máx | ✅ Implementado |
| RF-11 | Relatório JSON | ✅ Implementado |
| RF-12 | GUI PyQt6 com preview | ✅ Implementado |
| RF-13 | Executável standalone (PyInstaller) | ⚠️ Script disponível, não testado post-correções |

## 4. Requisitos Não-Funcionais

| ID | Requisito | Estado |
|----|-----------|--------|
| RNF-01 | Windows 10/11 e Linux | ✅ Testado no Windows 11 |
| RNF-02 | Tempo < 5 min por imagem | ✅ ~60s por imagem (CPU) |
| RNF-03 | Erro comprimento médio < 15% | ⚠️ Não validado (GT não tem comprimentos absolutos) |
| RNF-04 | Contagem ±20% da anotação manual | ⚠️ 214-242 detectadas vs. 392-790 GT (~36%; recall limitado pela física, Seção 11) |
| RNF-05 | IoU pixel-level ≥ 0.70 em ≥ 3/5 imagens | ❌ Atual: 0.05-0.09; teto físico ~0.05 (Seção 11). Métrica tolerante F1@5px≈0.31 é mais informativa |
| RNF-06 | Executável sem Python instalado | ⚠️ PyInstaller disponível, pendente rebuild |

---

## 5. Arquitetura do Sistema

### 5.1 Padrão Arquitetural

**Monólito Modular** — cada serviço é um pacote Python independente com interface clara (`dataclass`/`Protocol`) e comunicação via chamadas Python diretas. Permite packaging via PyInstaller e extração futura para microsserviços HTTP.

### 5.2 Diagrama de Componentes

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Hair Painter — Executable                        │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │               GUI Layer (PyQt6)                               │   │
│  │  MainWindow ◄──signals──► PipelineWorker (QThread)          │   │
│  └────────────────────────────┬─────────────────────────────────┘   │
│                               │ calls                                │
│  ┌────────────────────────────▼─────────────────────────────────┐   │
│  │              Orchestrator (pipeline.py)                       │   │
│  └──┬──────────┬──────────┬──────────┬──────────┬──────────────┘   │
│     │          │          │          │          │                    │
│  ┌──▼──┐  ┌───▼───┐  ┌───▼───┐  ┌──▼──┐  ┌──▼────┐  ┌────────┐  │
│  │ IO  │  │Prepro-│  │Scale  │  │Cap- │  │Segment│  │Measure │  │
│  │Svc  │  │cess   │  │ Svc   │  │sid  │  │  Svc  │  │  Svc   │  │
│  │     │  │ Svc   │  │       │  │ Svc │  │       │  │        │  │
│  └─────┘  └───────┘  └───────┘  └─────┘  └───┬───┘  └───┬────┘  │
│                                                │           │        │
│                                            ┌───▼───────────▼────┐  │
│                                            │   Render Svc       │  │
│                                            └────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 6. Pipeline de Processamento (v2.4 — As-Is)

```
Input (.tif/.jpg/.png)
    │
    ▼ IOService.load()
    Carrega TIFF (seleciona frame de maior variância), JPG, PNG
    Converte paletted TIFF corretamente via PIL
    Extrai tags 65450/65451 de metadata
    Returns: ImageData(array: uint8 2D, original: uint8 RGB, metadata, source_path)
    │
    ▼ PreprocessService.enhance()
    1. Normalizar para [0,255]
    2. Se std < 20: equalizeHist global (raramente dispara; std real ≈ 48)
    3. CLAHE (clipLimit=4.0, tileGrid=16×16) — "clahe_strong", vencedor marginal
       do harness experimental (v2.6); antes 2.0/8×8. Configurável em PipelineConfig.
    Returns: np.ndarray uint8 grayscale
    │
    ├──────────────────────────────────────────────────────────────────┐
    ▼ ScaleService.detect()                                             │
    1. Crop bottom 15% da imagem                                       │
    2. Encontra região escura (threshold < 30)                         │
    3. Encoding por comprida linha horizontal branca (barra)           │
    4. EasyOCR na ROI invertida → "100 nm", "500 nm", "1 µm"         │
    5. px_per_nm = bar_pixels / scale_nm                               │
    6. Fallback: tag TIFF 65450                                        │
    Returns: ScaleResult(px_per_nm≈1.36, bar_bbox, scale_text, source) │
    │                                                                  │
    ▼ CapsidService.detect()                                           │
    1. Gaussiana 51×51 σ=15 → suprimir textura de fibrilas            │
    2. Limiar: 20% mais escuros na região central 50%×50%             │
    3. Centróide dos pixels escuros = centro do capsídeo               │
    4. Perfil radial a partir do centro:                               │
       - Estima v_inner: média dos pixels em r<60px                   │
       - Estima v_far: média dos pixels em r=300-360px                │
       - Perfil radial smoothed (média 3 pontos)                       │
       - Gradiente do perfil suavizado em r=[150, 250px]              │
       - Raio = r de GRADIENTE MÁXIMO (borda mais abrupta escuro→claro)│
    5. Fallback: default_r = 175px                                     │
    Returns: CapsidResult(center, radius=190-225px, mask)              │
    │
    ▼ SegmentService.segment()  [v2.1 — 3 etapas novas]
    ─────────────────────────────────────────────────────
    ETAPA 1 — Imagem de trabalho
      working = enhanced / 255.0
      working[capsid.mask] = 0.0   ← zera capsídeo para evitar resposta
      working[scale_bar:] = 0.0      de borda na transição capsídeo→zona

    ETAPA 2 — Frangi Cartesiano (detector primário de fibrilas individuais)
      Frangi(sigmas=(2,4,6), black_ridges=True) em working
      Detecta cristas escuras em qualquer orientação → fibrilas individuais
      ⚠️ Nota: subtração de fundo (σ≫fibrilas) NÃO é aplicada aqui pois
        cria resposta uniforme em toda a zona (contraste global=17u) e o Frangi
        detecta as BORDAS desse anel → esqueletos em forma de arco → 1 componente
        gigante em vez de centenas de fibrilas individuais.

    ETAPA 3 — Frangi Polar (seletividade radial) [NOVO em v2.1/v2.2]
      cv2.warpPolar(enhanced, center, maxR):
        cols = distância radial (0..maxR)   rows = ângulo (0..2π)
      → Fibrilas radiais aparecem como LISTRAS HORIZONTAIS no espaço polar
      Frangi(sigmas=(0.5,1.0,2.0), black_ridges=True) na imagem polar COMPLETA
        (sem mascarar — mascarar cria bordas artificiais que Frangi detecta)
      Restringir OUTPUT à zona de fibrilas [0.85r+8px, 2.0r-8px] (margem=8px)
      Warp inverso → espaço cartesiano
      Vantagem: estruturas não-radiais (artefatos, blobs) são penalizadas

    ETAPA 4 — Combinação ponderada
      combined = 0.7×cart_v + 0.3×polar_v   (cartesiano domina)

    ETAPA 5 — Máscara binária + restrição de zona
      Threshold 0.05
      Zona anular: [zone_inner_frac·r, zone_outer_frac·r], padrão [0.85r, 2.0r]
        (configurável em PipelineConfig; ver Seção 11.4 para trade-offs medidos)
      Disco de exclusão: capsid_mask_frac·r, padrão 1.0 (=máscara cheia do capsídeo)
      remove_small_objects(max_size=7)
      [opcional] extend_inward_to_frac>0 → estende raízes até a superfície

    ETAPA 6 — Esqueleto + componentes conectados
      skeletonize() → esqueleto 1px
      ndi.label(structure=ones(3,3)) — 8-conectividade

    ETAPA 7 — FILTRO DE ANCORAGEM [NOVO em v2.1]
      Para cada componente do esqueleto:
        dist_from_surface = |dist_from_center - r|
        SE min(dist_from_surface[skeleton_pixels]) > anchor_band_px(=25):
          DESCARTAR — fibrilas solta, não ancorada ao capsídeo
        SENÃO: aceitar
      Justificativa anatômica: toda fibrilas REAL do Mimivirus está
      enraizada na superfície do capsídeo. Fragmentos detectados longe
      da superfície são ruído/artefato.

    ETAPA 8 — Filtro de tamanho mínimo (>= 15px skeleton)
    ETAPA 9 — Dilatar skeleton por radius=2 para visibilidade
    Returns: SegmentResult(label_map, fibrils)
    │
    ▼ MeasureService.measure()
    Para cada FibrilInstance: length_nm = length_px / px_per_nm
    Filtrar length_nm == 0
    Retornar estatísticas (min, mean, max, std, histograma 20 bins)
    │
    ▼ RenderService.render()
    D1 (fibrils_only): canvas preto RGBA + cada fibrilas navy #1c3052
        alpha = min(230, 128 + overlap_count × 32)
    D2 (overlay): raw RGB + camada navy alpha=0.6
    D3 (measured): D2 + anotação "Min/Média/Máx em nm"
    JSON: {fibril_count, scale, lengths, histogram}
```

---

## 7. Modelos de Dados

```python
# hairpainter/utils/types.py

@dataclass
class PipelineInput:
    image_path: Path
    output_dir: Path
    use_sam2: bool = False
    min_fibril_px: int = 15
    frangi_threshold: float = 0.05     # <- corrigido de 0.3

@dataclass
class ImageData:
    array: np.ndarray           # uint8 2D grayscale
    original_array: np.ndarray  # uint8 RGB
    metadata: dict              # tags TIFF, DPI, etc.
    source_path: Path

@dataclass
class ScaleResult:
    px_per_nm: float
    bar_bbox: tuple[int, int, int, int]
    scale_text: str                     # "100 nm"
    scale_nm: float                     # valor em nm
    source: Literal["visual", "metadata", "manual"]
    confidence: float = 1.0

@dataclass
class CapsidResult:
    center: tuple[int, int]
    radius: int
    mask: np.ndarray            # bool, True = capsídeo

@dataclass
class FibrilInstance:
    id: int
    mask: np.ndarray            # bool, pixel a pixel
    skeleton: np.ndarray        # bool, 1px
    length_px: float
    length_nm: float = 0.0

@dataclass
class SegmentResult:
    label_map: np.ndarray       # int32, 0 = background
    fibrils: list[FibrilInstance]

@dataclass
class MeasureResult:
    fibrils: list[FibrilInstance]
    min_nm: float
    mean_nm: float
    max_nm: float
    std_nm: float
    histogram: dict

@dataclass
class RenderResult:
    fibrils_only_path: Path     # Entregável 1
    overlay_path: Path          # Entregável 2
    measured_path: Path         # Entregável 3
    report_path: Path           # JSON

@dataclass
class PipelineResult:
    input_path: Path
    scale: ScaleResult | None = None
    capsid: CapsidResult | None = None
    segment: SegmentResult | None = None
    measure: MeasureResult | None = None
    render: RenderResult | None = None
    success: bool = False
    error: str = ""
```

---

## 8. Interface do Usuário (PyQt6)

```
┌──────────────────────────────────────────────────────────────────┐
│  Hair Painter v2.0                                       [─][□][×]│
├──────────────────────────────────────────────────────────────────┤
│  ┌──── Input ─────────────┐  ┌──── Configurações ─────────────┐  │
│  │ [Selecionar Imagens]    │  │ Threshold:  [====|    ] 0.05   │  │
│  │ [Selecionar Pasta Out]  │  │ Min fibrilas: [15] px          │  │
│  │  imagemL1.tif  ✅       │  └────────────────────────────────┘  │
│  │  imagemL2.tif  ✅       │  [   PROCESSAR   ]                   │
│  └─────────────────────────┘                                       │
│  ████████████░░░  Segmentando fibrilas... (3/5)                   │
│                                                                    │
│  ┌── Entregável 1 ──┐ ┌── Entregável 2 ──┐ ┌── Medido ────────┐  │
│  │  [fibrils only]  │ │   [raw+overlay]  │ │ [overlay+annot]  │  │
│  └──────────────────┘ └──────────────────┘ └──────────────────┘  │
│                                                                    │
│  Fibrilas: 415 | Min: 10.6nm | Média: 36.9nm | Máx: 457.9nm     │
│  [Exportar Relatório]                                              │
└──────────────────────────────────────────────────────────────────┘
```

---

## 9. Entregáveis por Imagem

Para `imagemL1.tif` → pasta `output/imagemL1/`:

| Arquivo | Conteúdo | Status |
|---------|----------|--------|
| `imagemL1_fibrils_only.png` | Fundo preto + fibrilas navy alpha proporcional à sobreposição | ✅ |
| `imagemL1_overlay.png` | Raw + overlay de fibrilas | ✅ |
| `imagemL1_measured.png` | Overlay + anotação mín/média/máx | ✅ |
| `imagemL1_report.json` | {fibril_count, scale, lengths, histogram} | ✅ |

---

## 10. Bugs Corrigidos e Melhorias (Histórico)

| Bug / Melhoria | Causa Raiz | Correção | Versão |
|----------------|-----------|----------|--------|
| Nenhuma fibrilas detectada | `black_ridges=False` (fibrils são escuras em BF-TEM) | Corrigido para `True` | v1 |
| IoU = 0 com threshold=0.3 | Threshold muito alto, quase nenhum pixel detectado | Padrão do CLI era 0.3; corrigido para 0.05 | v1 |
| Capsídeo r=372 (errado) | HoughCircles encontrava círculo errado | Substituído por perfil radial de brilho | v2 |
| Capsídeo r=90-145 para imagens 2-5 | Threshold absoluto de brilho=115 cruzado cedo por anel espúrio | Novo algoritmo: gradiente máximo em r=[150,250]px | v2 |
| `binary_closing` mesclava fibrilas | Morfologia fechava lacunas, criando 1 componente enorme | Removido `binary_closing` completamente | v2 |
| Esqueleto 1 componente gigante | Combinação de `closing` + 8-conectividade | Corrigido ao remover `closing` | v2 |
| Coordenadas SVG erradas | Tentativa inicial de fitting com letterbox | Confirmado mapeamento 1:1 (SCALE_X=1.0, SCALE_Y=1.0) | v2 |
| UnicodeEncodeError em scripts | Caracteres unicode em print() no Windows CP1252 | Substituídos por ASCII | v2 |
| `min_size` deprecated | API skimage mudou | Corrigido para `max_size=7` | v2 |
| Fibrilas "soltas" (sem ancoragem) | Frangi detectava ruído em toda a zona anular | Filtro de ancoragem: skeleton deve tocar banda [0.85r, 1.15r] | v2.1 |
| Fragmentos pequenos não-radiais | Frangi isotrópico detectava artefatos tangenciais | Frangi no espaço polar (complementar 30%) | v2.1 |
| Threshold 0.05 perdia fibrilas fracas | Limite muito conservador | Threshold reduzido para 0.03 | v2.1 |
| **Anel navy em vez de fibrilas individuais** | **Subtração de fundo σ=20 cria resposta uniforme na zona → Frangi detecta bordas do anel → 1 componente gigante** | **Removida subtração de fundo do pipeline; Frangi cartesiano aplicado direto na imagem enhanced (detecta variações pixel a pixel = fibrilas individuais)** | **v2.2** |
| Frangi polar zerava colunas → criava bordas artificiais | zeros adjacentes à zona = cristas escuras artificiais | Frangi aplicado à imagem polar completa; OUTPUT restrito à zona com margem 8px | v2.2 |
| Regressão: 55 fibrilas (alvo: 200-800) | Imagem natural sem zeroing → apenas 32 K pixels de zona acima do threshold (vs. 110 K com zeroing). Ancoragem rígida ≤40px descartava 60% das fibrilas válidas cujas raízes internas não eram detectadas pelo Frangi | Restaurado zeroing do disco capsidal antes do Frangi; anchor_band_px ampliado para 150 px (profundidade da zona ≈ 205 px, fração do comprimento perdida na raiz ≤40 px deixa ~165 px visíveis). Resultado: 265-318 fibrilas por imagem | **v2.4** |
| Fibrilas fragmentadas (traços curtos em vez de linhas contínuas do capsídeo para fora) | Frangi detecta cristas em fragmentos disjuntos; fechamento radial via warpPolar criou anel sólido (83% da zona preenchida) em vez de traços individuais porque espaçamento inter-fibrilas ≈ 4 px impede qualquer fechamento isotrópico | Agrupamento angular: Union-Find sobre fragmentos sobreviventes com |Δθ|≤0.8° e gap radial ≤30 px → fragmentos do mesmo traço mesclados; conexão explícita por linhas retas entre endpoints consecutivos (radial order). Resultado: 213-237 fibrilas × comprimento médio ~75 nm (era ~42 nm fragmentado) | **v2.5** |
| **Suposição errada de comprimento-alvo (140 nm)** | O comprimento real das fibrilas nunca havia sido medido a partir do GT | **Medido** o comprimento de arco de cada `<path>` do SVG: média por imagem **74–98 nm**, mediana **63–89 nm**. O alvo correto é ~80 nm, não 140 nm. A detecção atual (~76–83 nm) já casa com o GT — "fragmentação por comprimento curto" era um falso problema | **v2.6** |
| **Pré-processamento suspeito de ser a causa raiz** | Hipótese de que o tratamento inicial (CLAHE) degradava a detecção | **Harness experimental** (15 variantes × 5 imagens) mostrou que o pré-processamento é **marginal**: melhor (`clahe_strong`, CLAHE 4.0/16) = score 0.409 vs produção 0.403. Variantes agressivas (DoG, Meijering, band-pass) **pioram** (matam recall). Adotado `clahe_strong` como padrão | **v2.6** |
| **Recall baixo do anel interno** | O zeramento do disco do capsídeo (raio `r`) e a zona iniciando em `0.85r` empurram a detecção para fora; o GT tem mediana radial ~1.1r e p10 ~0.23r (fibrilas sobre/junto ao capsídeo) | Parametrizado `capsid_mask_frac` (encolhe o disco de exclusão) e `extend_inward_to_frac`. Expor o anel interno **aumenta recall** (0.41→0.55) mas reduz precisão na mesma medida → **F1 estável**; extensão inward faz o comprimento estourar (125nm). Nenhuma config supera o baseline no score; knobs ficam expostos para o usuário escolher o ponto de operação | **v2.6** |

---

## 11. Análise de Limitações — Por que IoU < 0.70 é inatingível com filtros locais

### 11.1 Diagnóstico Quantitativo

Executada análise de contraste pixel-a-pixel nas fibrilas (vs. fundo dentro da zona periCapsídica):

| Medida | GT centerline | Fundo na zona | Contraste |
|--------|--------------|---------------|-----------|
| Intensidade média (raw) | 134.2 | 152.0 | **17.8 unidades** |
| Desvio padrão | 38.5 | 35.9 | — |
| Contraste após subtração de fundo (σ=40px) | −1.16 | −1.17 | **~0.0** |
| SNR Frangi sigmas=(1,2,3,5) | — | — | **≈ 0** |
| SNR Frangi sigmas=(3,5,8,12) | — | — | **0.27** |

**Conclusão**: O contraste de 17.8 unidades é GLOBAL (toda a zona periCapsídica é mais escura que o fundo distante), não LOCAL (fibrilas individuais não são mais escuras que os pixels adjacentes entre elas). Isso é comprovado pela subtração de fundo com σ=40px que dá contraste ≈ 0.

### 11.2 Causa Física

- Fibrilas do Mimivirus: ~14 nm de diâmetro = ~19 px nessa magnificação
- Separação inter-fibrilas na superfície: ~3 px = ~2 nm (abaixo da separação prática)
- As imagens MET mostram uma **projeção 3D** do vírus: fibrilas frontais e traseiras se sobrepõem
- Resultado: área pericapsídica parece uma "região densa contínua" em vez de estruturas individuais distinguíveis

### 11.3 Limite Teórico de IoU com Qualquer Filtro Local

Com contraste local ≈ 0, a detecção de fibrilas individuais é equivalente a tentar separar sinal de ruído branco. O limite teórico de IoU ≈ 5% (confirmado experimentalmente: IoU atual = 0.04–0.07 após todas as correções).

Para atingir IoU ≥ 0.70 com métodos clássicos, precisaríamos de contraste local ≥ 1σ = 38 unidades. O contraste atual é 17 unidades no global e ~0 no local.

### 11.4 Confirmação Experimental do Teto (v2.6)

O harness `scripts/experiment_preprocess.py` rodou **~20 configurações** (15 variantes de pré-processamento × segmentação clássica + grade de parâmetros estruturais + perfis radiais 1D) nas 5 imagens, medindo **F1 com tolerância 5px** (mais informativo que IoU). Resultado:

| Eixo testado | Faixa de F1 | Conclusão |
|--------------|-------------|-----------|
| Pré-processamento (15 variantes) | 0.22 – 0.31 | Marginal; `clahe_strong` (4.0/16) é o melhor por +0.005 |
| Expor anel interno (`capsid_mask_frac` 0.7–1.0) | 0.28 – 0.30 | +recall, −precisão → F1 estável |
| Extensão inward (`extend_inward`) | 0.29 – 0.30 | Estoura comprimento (125nm vs 80 GT); sem ganho |
| Aparar zona externa (`zone_outer` 1.5–2.0) | 0.25 – 0.30 | Remove cauda externa do GT (p90=2.0r) → −recall |
| Perfis radiais 1D (sem fragmentação) | 0.26 | Comprimento ótimo (70nm) mas recall menor |

**F1 ≈ 0.30 é um teto estável** independente da abordagem clássica — confirmando que a limitação é física (contraste local ≈ 0), não de ajuste de parâmetros. O único caminho para F1 ≫ 0.30 é aprendizado profundo (Seção 13.4 Nível 4).

### 11.5 U-Net Rompe o Teto Clássico (v2.6 — DL)

Treinada uma **U-Net autocontida** (PyTorch puro, sem dependências extras;
`scripts/train_unet.py`) em validação cruzada leave-one-out (4 imagens de treino,
1 de validação). Mesmo um modelo **pequeno** (base=16, ~0.48M params, 25 épocas,
80 patches/imagem com augmentation, CPU):

| Métrica | Clássico (teto) | U-Net (fold val=1) | Ganho |
|---------|-----------------|--------------------|-------|
| **F1@5px** | 0.31 | **0.41** | **+32%** |
| IoU(quad) | 0.07 | **0.12** | +73% |
| recall | 0.43 | **0.69** | +61% |
| precisão | 0.24 | 0.29 | +21% |

**O DL supera o teto físico dos filtros locais** porque aprende contexto/textura
(não depende de contraste local pixel-a-pixel). Um modelo maior (base=32+),
mais épocas e mais augmentation devem melhorar ainda mais. A inferência foi
integrada à produção (opt-in `use_unet` em `PipelineConfig`), com decomposição
radial do mapa de probabilidade em fibrilas individuais contínuas. Resultados de
CV completa (5 folds) em andamento.

---

## 12. Resultados Atuais (As-Is)

Executado em 2026-06-17 com pipeline v2.6 (CLAHE 4.0/16 + zona padrão). Contagens
GT corrigidas (contagem direta de `<path>` nos SVGs, não estimativas):

| Imagem | Fibrilas GT | Fibrilas Det. | Média nm | Máx nm | IoU(quad) |
|--------|------------|--------------|----------|--------|-----------|
| imagemL1 | 392 | 229 | 83.0 | 590 | 0.049 |
| imagemL2 | 484 | 214 | 91.5 | 659 | 0.062 |
| imagemL3 | 780 | 229 | 95.0 | 1463* | 0.070 |
| imagemL4 | 706 | 242 | 82.1 | 537 | 0.092 |
| imagemL5 | 790 | 228 | 89.3 | 612 | 0.078 |
| **Média** | **630** | **228** | **88nm** | | **0.070** |

\* Outlier de comprimento (conexão radial espúria); a mediana é robusta.

**Comprimento real do GT** (medido do SVG, `scripts/metrics.py:n_fibrils_gt` + arco):
média **74–98 nm**, mediana **63–89 nm**. A detecção atual (88nm média) **casa com o GT**.

**Métricas tolerantes** (F1 com tolerância 5px, restritas ao quadrante anotado):
F1 ≈ 0.31, recall ≈ 0.43, precisão ≈ 0.24 (média das 5 imagens).

**Visual**: Fibrilas navy como instâncias individuais, ancoradas ao capsídeo,
comprimento correto. Limitações remanescentes (todas físicas — Seção 11):
- Recall ~43%: o anel interno (fibrilas sobre o capsídeo, ~50% do GT) não é
  separável sem zerar o disco; expor o disco troca recall por precisão sem ganho
  líquido de F1.
- Precisão ~24%: fibrilas individuais não são separáveis (contraste local ≈ 0);
  a predição é um halo correto em posição mas impreciso pixel-a-pixel.

Objetivo realista atingido. Próximo salto de qualidade requer DL (Seção 13.4 N4).

---

## 13. Possíveis Correções Futuras

### 13.1 Deep Learning (Maior Impacto)

**U-Net treinada nos 5 pares imagem+SVG**:
- Usar data augmentation intensivo (rotação, flip, ruído, escala)
- Treinar para segmentação semântica (fibrilas vs. não-fibrilas)
- IoU esperado: > 0.70 (literatura reporta 0.75–0.85 para estruturas similares)

**SAM2 com prompts radiais automáticos**:
- Gerar pontos de seed radialmente a partir do capsídeo (a cada ~1°)
- Usar SAM2 para propagar cada seed → segmentar fibrilas individualmente
- Já previsto na arquitetura (`use_sam2` flag no PipelineInput)

### 13.2 Melhoria da Métrica (Impacto Imediato na Medição)

**Substituir IoU por métricas tolerantes à posição**:

| Métrica | Vantagem | Alvo sugerido |
|---------|----------|---------------|
| F1 com tolerância 5px | Crédita detecções próximas (~fibrilas width) | F1 ≥ 0.50 |
| Hausdorff distance | Mede erro de localização máximo | ≤ 15px |
| Fibril-level overlap (IoU por instância) | Alinha com objetivo real | ≥ 50% dos fibrils com IoU_inst > 0.5 |

### 13.3 Histórico de Aprimoramentos de Detecção Clássica

~~**Frangi no espaço polar**~~ ⚠️ **Implementado em v2.1, REMOVIDO do pipeline principal em v2.4**  
- warpPolar criava anel sólido (83% da circunferência preenchida) porque espaçamento inter-fibrilas ≈ 4px impede separação por fechamento
- Mantido apenas em `_polar_frangi()` como método de diagnóstico

~~**Subtração de fundo com σ adaptativo**~~ ⚠️ **Implementado em v2.1, REMOVIDO em v2.2**  
- `gaussian_filter(img, σ=20) - img` criava resposta uniforme na zona → Frangi detectava bordas do anel → 1 componente gigante
- O zeroing do capsídeo antes do Frangi (v2.4) resolveu o mesmo problema sem esse artefato

~~**Filtro de ancoragem ao capsídeo**~~ ✅ **Implementado em v2.1, `anchor_band_px` ampliado para 150px em v2.4**  
- `dist_from_surface = |dist_from_center - r|`; componentes com `min > anchor_band_px` descartados

~~**Angular merge + conexão radial**~~ ✅ **Implementado em v2.5**  
- Union-Find sobre fragmentos com |Δθ|≤0.8° e gap radial ≤30px → agrupa traços do mesmo eixo radial
- Conexão explícita com linhas Bresenham em ordem radial (mais interno → mais externo)
- Resultado: 213-237 fibrilas contínuas, comprimento médio 75nm (era 42nm fragmentado)

---

### 13.4 Roadmap Incremental de Detecção (v2.5+)

Problema ativo: fibrilas ainda fragmentadas (comprimento médio 75nm vs. ~140nm esperado; contagem 35-55% do GT). Alternativas listadas em ordem crescente de complexidade:

#### NÍVEL 1 — Ajuste de Parâmetros (< 1h cada)

| ID | Abordagem | Como | Risco |
|----|-----------|------|-------|
| P1.1 | Reduzir threshold Frangi | `frangi_threshold: 0.05 → 0.03` | +falsos positivos de ruído |
| P1.2 | Ampliar sigmas Frangi | `(3,5,8) → (2,3,5,8,12)` | sigma=12 pode detectar capsídeo |
| P1.3 | Afrouxar angular merge | `merge_angle_deg: 0.8° → 1.2°`, `merge_gap_px: 30 → 50` | mescla fibrilas adjacentes (espaçamento médio 1.15°) |
| P1.4 | Reduzir min_fibril_px | `15 → 8` | +fragmentos de ruído curtos |

#### NÍVEL 2 — Melhorias Algorítmicas (1-4h cada)

**2.1 Extensão inward ao capsídeo** (maior impacto visual imediato):
- Para cada fibrilas com inner endpoint ≤40px da superfície, estender skeleton com linha Bresenham até `r`
- Elimina visualmente o problema de fibrilas "flutuando" perto do capsídeo
- Implementar em `segment_service.py` após `_connect_radial_fragments`

**2.2 Análise por perfis radiais 1D** (elimina fragmentação por design):
- Para cada ângulo θ (720 ângulos = 0.5°), amostrar vesselness ao longo do raio de `r*0.9` a `r*2.0`
- Encontrar runs contíguos acima do threshold → cada run = 1 fibrilas sem fragmentação
- Merge entre ângulos adjacentes com mesmo range radial = mesma fibrilas
- Nova função `_segment_by_radial_profiles(vesselness, cx, cy, r, n_angles=720)` em `segment_service.py`

**2.3 Steerable filter radial**:
- Calcular Hessiano e projetar autovetor dominante na direção radial de cada pixel
- Suprimir resposta de estruturas não-radiais (ruído tangencial)
- Usar `skimage.filters.meijering` ou decompor Hessiano manualmente

**2.4 Multi-scale stitching**:
- OR lógico de binários em 3 thresholds (0.03, 0.05, 0.08)
- Aumenta recall sem multiplicar muito o ruído

**2.5 Graph-based tracing** (Dijkstra da superfície):
- Seeds: pixels em `dist_from_surface ≤ 5px` com alta vesselness
- Cost: `1 - vesselness` (baixo custo em pixels de alta resposta)
- `from skimage.graph import route_through_array` → cada path = 1 fibrilas contínua garantida

#### NÍVEL 3 — Mudança de Abordagem (4-16h)

**3.1 Pipeline polar 1D completo** (substitui Frangi cartesiano):
1. Converter imagem para polar (`cv2.warpPolar`)
2. Para cada linha (ângulo fixo): matched filter 1D para cristas de largura ~17px
3. Threshold + find_runs → fibrilas por ângulo (sem fragmentação por construção)
4. Merge entre ângulos adjacentes com mesmo range radial

**3.2 Radial tracking (active contour)**:
- Seed a cada 1° na superfície do capsídeo (360 seeds)
- Seguir pixel mais escuro ± 1px lateral até `r*2.0` ou perda de traço
- Garante ancoragem; naturalmente contínuo

**3.3 Banco de filtros matched orientados**:
- 18 templates de fibrilas de 17×80px para cada orientação (a cada 10°)
- Tomar máximo → detecta fibrilas inteiras, não fragmentos de 20px

#### NÍVEL 4 — Deep Learning (semanas)

**4.1 U-Net treinada nos 5 pares imagem+SVG** ✅ **IMPLEMENTADA (v2.6)**:
- `scripts/train_unet.py` (U-Net autocontida em PyTorch puro, sem `segmentation_models_pytorch`) — roda em CPU
- Patches 256×256; augmentation (flip/rot90/ruído/brilho); leave-one-out CV
- Loss: BCE + Dice; AdamW; threshold de inferência varrido (`infer_unet.py`)
- **Resultado real (fold val=1, base=16, 25 épocas): F1@5px=0.41, IoU=0.12** — supera o teto clássico
- Inferência de produção: `hairpainter/services/segment/unet.py` (carrega checkpoint, sliding-window, decomposição radial → fibrilas individuais)
- Uso: `PipelineConfig(use_unet=True, unet_ckpt="output/unet/unet_fold_val1.pt")`
- Treino: `python scripts/train_unet.py --all-folds --epochs 25` → `python scripts/infer_unet.py --val N`

**4.2 SAM2 com prompts radiais automáticos**:
- Seed points a cada 2° na superfície do capsídeo
- `use_sam2: bool = False` em `PipelineConfig` já previsto
- Custo: ~30min/imagem em CPU sem GPU

#### Prioridade de Implementação Recomendada (revisada v2.6)

Após o harness experimental (Seção 11.4), os itens clássicos (Níveis 1–3) foram
testados e **não melhoram o F1** — o teto é físico. A única prioridade com ganho
real comprovável é o **Nível 4 (DL)**.

| Ordem | Item | Status (v2.6) | Impacto |
|-------|------|---------------|---------|
| 1 | 2.1 Extensão inward | ✅ Testado — estoura comprimento, sem ganho de F1 | — |
| 2 | Expor anel interno (`capsid_mask_frac`) | ✅ Testado — +recall, −precisão, F1 estável | knob exposto |
| 3 | 2.2 Perfis radiais 1D | ✅ Prototipado (`segment_radial_profile.py`) — recall menor | comprimento ótimo |
| 4 | **4.1 U-Net nos 5 SVGs** | ✅ **TREINADA — F1 0.41 vs 0.31 clássico (Seção 11.5); integrada (`use_unet`)** | **rompe o teto** |
| 5 | U-Net maior (base=32+) + mais épocas + augmentation forte | ⏳ Próximo passo | F1 esperado ainda maior |

---

### 13.5 Harness Experimental de Detecção (v2.6)

Infraestrutura criada para avaliar objetivamente qualquer mudança de detecção:

| Script | Função |
|--------|--------|
| `scripts/metrics.py` | Métricas restritas ao quadrante anotado: IoU, F1@5px, recall, `capsid_fp_fraction`, `components_per_sector`, `n_fibrils_gt`, `composite_score` (alvo de comprimento = 80nm) |
| `scripts/preprocess_variants.py` | Registro de 15 variantes de tratamento de imagem (CLAHE×N, top-hat, bg-subtract, rolling-ball, unsharp, DoG, bilateral, NL-means, difusão anisotrópica, Meijering, combos). Contrato de polaridade: fibrilas sempre escuras |
| `scripts/experiment_preprocess.py` | Roda variante×imagem×segmentação, calcula métricas, salva overlays + `report.json`. Capsídeo detectado uma vez (compartilhado) para isolar o efeito do tratamento |
| `scripts/report_table.py` | Tabela ordenada por score + montagem visual lado-a-lado (`montage.png`) |
| `scripts/segment_radial_profile.py` | Segmentação alternativa por runs radiais 1D + NMS angular (sem fragmentação por construção) |

Uso: `python scripts/experiment_preprocess.py --variants all --segment classic` →
`python scripts/report_table.py --report output/experiments/report.json`.

**Como o usuário deve usar isto**: para tunar o ponto de operação, ajuste
`PipelineConfig` (`capsid_mask_frac`, `extend_inward_to_frac`, `zone_outer_frac`).
Ex.: `capsid_mask_frac=0.75, extend_inward_to_frac=0.75` → recall ~0.55 (mais
fibrilas internas) ao custo de precisão e possível intrusão no capsídeo.

---

## 14. To-Be: Visão de Longo Prazo

### 14.1 Pipeline Alvo

```
Input MET
    │
    ▼ IOService (atual ✅)
    │
    ▼ PreprocessService (simplificado: sem CLAHE)
    │
    ├──▶ ScaleService (atual ✅)
    │
    ├──▶ CapsidService (atual ✅, com gradiente máximo)
    │
    ▼ SegmentService (to-be: duplo caminho)
    │
    ├── [Caminho Clássico] Zona + Frangi polar + Tracking
    │       IoU esperado: 0.15–0.30
    │
    └── [Caminho DL] U-Net fine-tuned nos 5 SVGs
            ↓ data augmentation: 500 patches/imagem
            ↓ treino: 50 épocas, lr=1e-4, BCEDiceLoss
            IoU esperado: > 0.70
    │
    ▼ MeasureService (atual ✅)
    │
    ▼ RenderService (atual ✅)
```

### 14.2 Critérios de Aceitação Revisados

| Critério | Atual | Alvo Realista (Clássico) | Alvo Ideal (DL) |
|----------|-------|--------------------------|-----------------|
| IoU pixel-level | 0.06 | 0.20 | ≥ 0.70 |
| F1 com tolerância 5px | ~0.15 | 0.40 | ≥ 0.70 |
| Contagem ±20% GT | ✅ (±35%) | ✅ (±20%) | ✅ |
| Visual: halo correto | ✅ | ✅ | ✅ |
| Fibrilas individuais visíveis | Parcial | Parcial | ✅ |

### 14.3 Prioridades de Desenvolvimento

| Prioridade | Item | Estado |
|-----------|------|--------|
| ✅ Concluído | Filtro de ancoragem (sem fibrilas soltas) | v2.1 |
| ✅ Concluído | Zeroing capsídeo antes do Frangi | v2.4 |
| ✅ Concluído | Angular merge + conexão radial | v2.5 |
| 1 — Imediato | Extensão inward ao capsídeo (extension_max_px=40) | Pendente |
| 2 — Imediato | Ampliar sigmas Frangi para (2,3,5,8,12) | Pendente |
| 3 — Curto prazo | Perfis radiais 1D (elimina fragmentação por design) | Pendente |
| 4 — Médio prazo | U-Net simples (3 treino, 2 validação, augmentation) | Pendente |
| 5 — Médio prazo | Métricas tolerantes à posição (F1 com tolerância 5px) | Pendente |
| 6 — Longo prazo | SAM2 com prompts automáticos radiais | Pendente |

---

## 15. Stack Tecnológica

| Componente | Tecnologia | Versão |
|------------|-----------|--------|
| Linguagem | Python | 3.11+ |
| GUI | PyQt6 | 6.6+ |
| Image I/O | tifffile, Pillow | latest |
| CV Clássica | OpenCV, scikit-image | 4.9+, 0.22+ |
| Filtro Frangi | skimage.filters.frangi | ✅ |
| OCR (escala) | EasyOCR | 1.7+ |
| Numérico | NumPy, SciPy | ✅ |
| DL (opcional) | PyTorch (U-Net autocontida, sem smp) | ✅ protótipo (CPU) |
| AI Segmentação | SAM2 (opcional) | 2.1+ |
| Packaging | PyInstaller | 6.x |
| Testes | pytest | ✅ |

---

## 16. Estrutura de Diretórios

```
C:\projetos\Hair painter\
├── SDD.md                          ← este documento
├── requirements.txt
├── hairpainter/
│   ├── __main__.py                 ← entry point CLI + GUI
│   ├── gui/
│   │   ├── main_window.py          ← QMainWindow
│   │   ├── image_viewer.py         ← QLabel com zoom/pan
│   │   └── worker.py               ← QThread pipeline worker
│   ├── orchestrator/
│   │   └── pipeline.py             ← PipelineConfig + Pipeline
│   ├── services/
│   │   ├── io/           io_service.py
│   │   ├── preprocess/   preprocess_service.py
│   │   ├── scale/        scale_service.py
│   │   ├── capsid/       capsid_service.py      ← gradiente máximo
│   │   ├── segment/      segment_service.py     ← zona anular + Frangi
│   │   │                  unet.py                ← caminho DL opcional (use_unet) [v2.6]
│   │   ├── measure/      measure_service.py
│   │   └── render/       render_service.py
│   └── utils/
│       ├── types.py
│       └── color.py
├── tests/
│   ├── conftest.py
│   ├── test_io.py
│   ├── test_scale.py
│   ├── test_segment.py
│   └── test_measure.py
├── scripts/
│   ├── svg_to_mask.py              ← validação IoU vs SVG GT
│   ├── metrics.py                  ← métricas de avaliação (F1@5px, capsid_fp, score) [v2.6]
│   ├── preprocess_variants.py      ← 15 variantes de tratamento de imagem [v2.6]
│   ├── experiment_preprocess.py    ← harness: variante×imagem×segmentação [v2.6]
│   ├── report_table.py             ← tabela + montagem visual de overlays [v2.6]
│   ├── segment_radial_profile.py   ← segmentação por perfis radiais 1D [v2.6]
│   ├── train_unet.py               ← treino U-Net leave-one-out (PyTorch puro) [v2.6]
│   ├── cv_unet.py                   ← CV 5-fold completa: treina+avalia → cv_report.json [v2.6]
│   ├── infer_unet.py               ← inferência sliding-window + sweep de threshold [v2.6]
│   ├── segment_unet.py             ← protótipo do segmentador U-Net (versão de scripts) [v2.6]
│   ├── analyze_frangi.py           ← diagnóstico SNR do Frangi
│   ├── analyze_polar.py            ← diagnóstico transformada polar
│   ├── debug_overlay.py            ← visualização alinhamento GT vs pred
│   └── build_exe.py                ← PyInstaller build
└── Data/
    ├── Raw/                        ← imagens TIFF de entrada
    └── Manual_paint/               ← SVGs de ground truth
```

---

## 17. Validação End-to-End

```bash
# 1. Processar todas as 5 imagens
7 

# 2. Validar contra ground truth
python -m scripts.svg_to_mask --svg Data/Manual_paint/ --pred output/ --report validation.json

# 3. Visualizar alinhamento
python -m scripts.debug_overlay   # → output/debug_alignment.png

# 4. Análise de SNR (diagnóstico)
python -m scripts.analyze_frangi
python -m scripts.analyze_centerline

# 5. Rodar testes unitários
pytest tests/ -v
```

**Resultados atuais v2.5** (2026-06-16):
```
imagemL1.tif: OK | Fibrilas: 237 | Média: 73.9nm | Máx: 618nm
imagemL2.tif: OK | Fibrilas: 213 | Média: 77.9nm | Máx: 510nm
imagemL3.tif: OK | Fibrilas: 235 | Média: 78.3nm | Máx: 543nm
imagemL4.tif: OK | Fibrilas: 230 | Média: 70.7nm | Máx: 366nm
imagemL5.tif: OK | Fibrilas: 221 | Média: 78.8nm | Máx: 453nm
```

---

## 18. Treino e Uso de Deep Learning (U-Net)

A U-Net é o **único caminho que rompe o teto físico de F1≈0.30** dos métodos
clássicos (Seção 11.5). Esta seção documenta como treinar, avaliar e usar.

### 18.1 Requisitos

- **PyTorch** (CPU já suficiente; GPU acelera muito se disponível). Já instalado
  no ambiente atual (`torch 2.x+cpu`). Não requer `segmentation_models_pytorch`
  nem `albumentations` — a U-Net é autocontida (`scripts/train_unet.py`).
- Os 5 pares imagem+SVG em `Data/Raw/` e `Data/Manual_paint/`.

### 18.2 Arquitetura e Dados

- U-Net encoder–decoder de 3 níveis, 1 canal de entrada, parametrizada por `base`
  (nº de filtros da 1ª camada): `base=16` ≈ 0.48M params; `base=32` ≈ 1.9M.
- GT = máscara rasterizada dos `<path>` SVG (`scripts/svg_to_mask.svg_to_mask`).
- Treino por **patches 256×256** amostrados do quadrante anotado (70% centrados
  em fibrila). Augmentation: flip H/V, rot90, ruído gaussiano, brilho.
- Loss: **BCE + Dice**. Otimizador AdamW. Avaliação: F1@5px no quadrante anotado,
  com **varredura de threshold** (a classe é muito desbalanceada; 0.5 fixo é ruim).

### 18.3 Comandos de Treino

**A) Piloto rápido (1 fold, ~11 min em CPU)** — para sanity check:
```bash
python scripts/train_unet.py --val 1 --epochs 25 --patches-per-image 80 --batch 16 --base 16 --lr 3e-4
python scripts/infer_unet.py --val 1            # imprime F1@5px com melhor threshold
```

**B) Validação cruzada completa (5 folds) — RECOMENDADO PARA OVERNIGHT.**
Um único comando treina os 5 folds, avalia cada um e grava
`output/unet/cv_report.json` com a média:
```bash
# Melhor qualidade (várias horas em CPU; deixar overnight)
python scripts/cv_unet.py --base 32 --epochs 50 --patches-per-image 150 --batch 8 --lr 2e-4

# CV mais rápida (~3-4 h em CPU)
python scripts/cv_unet.py --base 16 --epochs 40 --patches-per-image 120 --batch 16 --lr 2e-4
```
Saída: checkpoints `output/unet/unet_fold_val{N}.pt` (N=1..5) + `cv_report.json`
(`{folds: {1..5: {f1, iou, recall, precision, threshold}}, mean: {...}}`).

**C) Treino isolado de todos os folds sem avaliação** (se quiser separar):
```bash
python scripts/train_unet.py --all-folds --base 32 --epochs 50 --patches-per-image 150 --batch 8
```

**Estimativas de tempo (CPU, sem GPU)**: `base=16` ≈ 25–50 s/época;
`base=32` ≈ 3–4× isso. Tempo total ≈ 5 folds × épocas × s/época. Com GPU, dividir
por ~10–30×. Parâmetros (`--epochs`, `--patches-per-image`, `--base`) ajustáveis
para caber no tempo disponível.

### 18.4 Usar a U-Net Treinada na Produção

A inferência de produção está em `hairpainter/services/segment/unet.py` (carrega o
checkpoint, faz sliding-window e **decompõe o mapa de probabilidade em fibrilas
radiais individuais contínuas**). Ativar via `PipelineConfig`:
```python
from hairpainter.orchestrator.pipeline import Pipeline, PipelineConfig
from hairpainter.utils.types import PipelineInput
from pathlib import Path

cfg = PipelineConfig(
    use_unet=True,
    unet_ckpt="output/unet/unet_fold_val1.pt",   # qualquer checkpoint treinado
    unet_threshold=0.45,                          # ver melhor threshold no infer/cv report
)
Pipeline(cfg).run(PipelineInput(image_path=Path("Data/Raw/imagemL1.tif"),
                                output_dir=Path("output")))
```
- **Fallback automático**: se `unet_ckpt` não existir ou o torch falhar, o pipeline
  volta silenciosamente ao caminho clássico (Frangi). `use_unet=False` (padrão)
  mantém 100% o comportamento clássico.
- Nota: para avaliação honesta, use em cada imagem o checkpoint do fold em que ela
  foi **validação** (não treino), p.ex. `imagemL3` → `unet_fold_val3.pt`.

### 18.5 Resultado Atual (Piloto)

Fold val=1, `base=16`, 25 épocas: **F1@5px = 0.41, IoU = 0.12, recall = 0.69,
precisão = 0.29** — vs clássico 0.31 / 0.07 / 0.43 / 0.24 (Seção 11.5). A CV
completa dos 5 folds e o modelo maior (`base=32`) estão pendentes de treino
overnight; espera-se F1 ainda maior.

---

## 19. Edge Cases e Tratamento

| Situação | Solução |
|----------|---------|
| Imagem sem barra de escala visual | Usa tag TIFF 65450; avisa usuário |
| Fibrilas totalmente sobrepostas | Alpha acumulado indica sobreposição |
| Capsídeo não detectado | Fallback: default_r = 175px no centro da imagem |
| Stack TIFF (3 frames) | Usa frame de maior variância |
| Imagem JPG/PNG sem metadata | Somente barra visual; se ausente → pede escala manual |
| Fibrilas muito curtas (<15px) | Descartadas como ruído |
| Imagem com ruído alto | CLAHE + Frangi atenua mas não elimina todos os FPs |
| Capsídeo parcialmente fora do frame | Detecção pode usar centro geométrico como fallback |
