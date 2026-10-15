# MistralFit — avvio rapido

Questo progetto cerca una configurazione di llama.cpp adatta al tuo modello Mistral confrontando prove reali. La prima versione contiene il programma, i test e la documentazione per GitHub. **Le prestazioni su Mistral Small 4 completo restano da misurare.**

## 1. Provalo subito

Estrai lo ZIP, apri PowerShell nella cartella che contiene questo file ed esegui:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\Start-MistralFit.ps1
```

Lo script cerca Python 3.10 o successivo, mostra RAM e GPU disponibili e stampa un esempio di piano di memoria. L'esempio usa dimensioni inventate e dichiarate: non scarica un modello e non simula una velocità di inferenza.

Se manca Python, installalo con:

```powershell
winget install -e --id Python.Python.3.13 --accept-source-agreements --accept-package-agreements
```

Poi riapri PowerShell e riesegui lo script. Per i comandi Python puoi usare `py -3`; se il tuo PC ha solo `python`, sostituisci `py -3` con `python`.

## 2. Prepara una prova reale

Servono due elementi esterni al pacchetto:

1. `llama-server.exe`, da una [build ufficiale di llama.cpp](https://github.com/ggml-org/llama.cpp/releases) compatibile con la tua GPU e con Small 4.
2. Un GGUF di Mistral Small 4 che abbia dimensioni compatibili con RAM e GPU. La [scheda ufficiale del modello](https://huggingface.co/mistralai/Mistral-Small-4-119B-2603) rimanda alle distribuzioni supportate.

Il programma non compra servizi, non usa API cloud e non scarica automaticamente decine di GB di pesi. Se il modello è diviso in più file, tienili tutti nella stessa cartella e indica il primo.

```powershell
py -3 -m mistralfit doctor --server "C:\llama.cpp\llama-server.exe"
py -3 -m mistralfit plan "D:\Models\modello.gguf" --ctx 4096
py -3 -m mistralfit tune "D:\Models\modello.gguf" `
  --server "C:\llama.cpp\llama-server.exe" --ctx 4096 --tokens 128 --repeats 3
```

Trovi il risultato in `results\experiment.json` e il report da aprire nel browser in `results\experiment.html`. La prova carica più volte il modello: può richiedere parecchio tempo. Con Ctrl+C interrompi e il processo del benchmark viene fermato.

## 3. Avvia la configurazione misurata

```powershell
py -3 -m mistralfit serve "D:\Models\modello.gguf" `
  --server "C:\llama.cpp\llama-server.exe" --report .\results\experiment.json
```

Apri `http://127.0.0.1:8000`. Le API locali sono disponibili su `http://127.0.0.1:8000/v1`.

## 4. Sviluppa e contribuisci

Esegui prima i test:

```powershell
py -3 -m unittest discover -v
```

La repository ufficiale del progetto è [Berserk-hub150/MistralFit](https://github.com/Berserk-hub150/MistralFit). Per contribuire, leggi [CONTRIBUTING.md](CONTRIBUTING.md), apri un fork e proponi una modifica piccola con un caso riproducibile. Puoi condividere un esperimento reale tramite il modello di issue dedicato.

Lo script `scripts\Publish-GitHub.ps1` rimane disponibile per chi parte da uno ZIP e desidera creare una propria repository separata. Richiede il parametro esplicito `-Publish`; non va usato per aggiornare la repository esistente.

Usa questa prima versione come progetto sperimentale e invito a contribuire. Prima di promuoverlo come soluzione più veloce, pubblica almeno un confronto reale e ripetibile con `upstream-auto` sulla stessa macchina. Nel file [piano di lancio](docs/launch-plan.md) trovi la sequenza consigliata.
