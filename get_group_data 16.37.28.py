import pandas as pd
import shutil
from pathlib import Path

# --- CONFIGURAZIONE ---
GROUP_ID = "a" 
CSV_NAME = "data_with_splits copy.csv"

# Percorsi Sorgente (dove si trovano i tuoi file ora)
PATH_DATA = Path("./data")
PATH_IMGS_SRC = PATH_DATA / "img"   # Cartella sorgente immagini
PATH_MASKS_SRC = PATH_DATA / "masks"  # Cartella sorgente maschere

# Percorsi Destinazione (dove vuoi copiare i file del gruppo A)
PATH_PROCESSED = PATH_DATA / "group_data"
PATH_IMGS_OUT = PATH_PROCESSED / "imgs"
PATH_MASKS_OUT = PATH_PROCESSED / "masks"

# 1. Creazione cartelle di destinazione
PATH_IMGS_OUT.mkdir(parents=True, exist_ok=True)
PATH_MASKS_OUT.mkdir(parents=True, exist_ok=True)

# 2. Caricamento CSV
if not Path(CSV_NAME).exists():
    print(f"ERRORE: Non trovo il file {CSV_NAME}")
else:
    df_labels = pd.read_csv(CSV_NAME)
    
    # Pulizia nomi colonne e filtraggio per GROUP_ID
    df_labels.columns = df_labels.columns.str.strip()
    df_labels_group = df_labels[df_labels["group_id"].astype(str).str.strip() == str(GROUP_ID)]

    print(f"Righe trovate nel CSV per il gruppo {GROUP_ID}: {len(df_labels_group)}")

    if df_labels_group.empty:
        print("ATTENZIONE: Il CSV filtrato è vuoto. Controlla che i nomi dei gruppi siano corretti.")
    else:
        # 3. Ciclo di copia
        count = 0
        for _, row in df_labels_group.iterrows():
            img_name = row["img_id"]
            # Gestione nome maschera: se img è '1.png', la maschera è '1_mask.png'
            mask_name = img_name.replace(".png", "_mask.png")
            
            # Costruzione percorsi assoluti/relativi corretti
            src_img = PATH_IMGS_SRC / img_name
            src_mask = PATH_MASKS_SRC / mask_name
            
            dst_img = PATH_IMGS_OUT / img_name
            dst_mask = PATH_MASKS_OUT / mask_name
            
            try:
                # Copia file immagine
                shutil.copyfile(src_img, dst_img)
                # Copia file maschera
                shutil.copyfile(src_mask, dst_mask)
                count += 1
            except FileNotFoundError:
                print(f"MANCANTE: Non trovo {src_img.name} in {PATH_IMGS_SRC} o la sua maschera.")
            except Exception as e:
                print(f"ERRORE su {img_name}: {e}")

        # 4. Salvataggio del nuovo CSV filtrato
        output_csv = PATH_PROCESSED / "metadata.csv"
        df_labels_group.to_csv(output_csv, index=False)
        
        print("-" * 30)
        print(f"FINE! Copiate {count} coppie di immagini/maschere.")
        print(f"Il nuovo CSV si trova in: {output_csv}")

        import pandas as pd
from pathlib import Path

# Configurazione percorsi
PATH_DATA = Path("./data")
PATH_IMGS_SRC = PATH_DATA / "img"
PATH_MASKS_SRC = PATH_DATA / "mask"
CSV_NAME = "data_with_splits copy.csv"

print(f"--- TEST DI DIAGNOSI ---")

# 1. Verifica esistenza cartelle
print(f"Cartella immagini esiste? {PATH_IMGS_SRC.exists()} ({PATH_IMGS_SRC.absolute()})")
print(f"Cartella maschere esiste? {PATH_MASKS_SRC.exists()} ({PATH_MASKS_SRC.absolute()})")

# 2. Leggi i primi file reali nelle cartelle (se esistono)
if PATH_IMGS_SRC.exists():
    files_reali = list(PATH_IMGS_SRC.glob("*"))[:3]
    print(f"Esempio file reali in /img: {[f.name for f in files_reali]}")

# 3. Carica il CSV e controlla i dati
if Path(CSV_NAME).exists():
    df = pd.read_csv(CSV_NAME)
    df.columns = df.columns.str.strip()
    
    # Prendi la prima riga del gruppo A
    riga_test = df[df["group_id"].astype(str).str.strip() == "A"].head(1)
    
    if not riga_test.empty:
        nome_csv = riga_test["img_id"].values[0]
        print(f"Nome file cercato dal CSV: '{nome_csv}'")
        
        # Verifica se quel file esiste davvero
        percorso_test = PATH_IMGS_SRC / nome_csv
        print(f"Il computer lo cerca in: {percorso_test}")
        print(f"IL FILE ESISTE SUL DISCO? {'✅ SI' if percorso_test.exists() else '❌ NO'}")
    else:
        print("❌ Non ho trovato nessuna riga con group_id = 'A'")
else:
    print(f"❌ Il file {CSV_NAME} non esiste.")