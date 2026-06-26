# Como subir o projeto para o Kaggle

## Pré-requisitos

```powershell
pip install kaggle
```

Coloque suas credenciais em `%USERPROFILE%\.kaggle\kaggle.json`:
```json
{"username": "dias0202", "key": "SEU_API_KEY"}
```
(API key em: https://www.kaggle.com/settings → API → Create New Token)

---

## Passo 1 — Fazer upload do dataset (código + dados)

A partir da raiz do projeto (`C:\projetos\Hair painter`):

```powershell
cd "C:\projetos\Hair painter"

# Primeira vez: cria o dataset no Kaggle
kaggle datasets create -p . --dir-mode tar

# Atualizações futuras:
kaggle datasets version -p . -m "descrição da mudança" --dir-mode tar
```

O dataset ficará em: https://www.kaggle.com/datasets/dias0202/hair-painter

> **Nota**: O `.kaggleignore` na raiz exclui automaticamente `.git/`, `output/`, `__pycache__/`, etc.

---

## Passo 2 — Criar o notebook no Kaggle

### Opção A — Upload do .ipynb (mais rápido)
1. Acesse https://www.kaggle.com/YOUR_USERNAME/notebooks
2. "New Notebook" → "File" → "Upload Notebook"
3. Selecione `kaggle/hair_painter_train.ipynb`
4. Clique em "Add Data" → "Your Datasets" → `hair-painter`

### Opção B — Via API
```powershell
# Empurra o notebook para o Kaggle (requer notebook-metadata.json)
kaggle kernels push -p kaggle/
```

---

## Passo 3 — Configurar acelerador

No notebook Kaggle:
- Clique em **Settings** (lado direito)  
- **Accelerator**: escolha `GPU T4 x2` (recomendado)
- **Internet**: pode deixar desligado (não precisa de download externo)

---

## Passo 4 — Rodar o notebook

- Clique **"Run All"** (ou Ctrl+Shift+F9)
- O treino de 19 folds × 150 épocas demora ~4-6h com T4 x2
- Progresso aparece no output de cada célula

---

## Passo 5 — Baixar os checkpoints

Ao fim do treino:
1. Clique em **"Output"** na barra lateral
2. Navegue até `output/unet/`
3. Baixe todos os `unet_fold_val*.pt`

Ou via API:
```powershell
# Lista outputs
kaggle kernels output NOTEBOOK_ID

# Baixa para pasta local
kaggle kernels output NOTEBOOK_ID -p output/unet_kaggle/
```

---

## Passo 6 — Usar os checkpoints localmente

Coloque os `.pt` baixados em `output/unet/` e rode:

```powershell
python scripts/infer_unet.py --val 1 --ckpt output/unet/unet_fold_val1.pt
python scripts/cv_unet.py --start-fold 1  # avalia todos os folds
```

---

## Aceleradores disponíveis e configs recomendadas

| Acelerador | epochs | batch | base | patches/img | tempo est. |
|------------|--------|-------|------|-------------|------------|
| T4 x2 ✓   | 150    | 64    | 64   | 400         | ~4-6 h     |
| T4 x1      | 150    | 32    | 64   | 400         | ~7-9 h     |
| P100       | 150    | 32    | 64   | 400         | ~6-8 h     |
| TPU v3-8   | 150    | 64    | 64   | 400         | ~3-4 h     |

Para testar com menos tempo, use no notebook: `all_folds=False, epochs=10, base=16`.
