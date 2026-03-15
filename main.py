import os
import torch
from torch.utils.data import DataLoader

from src.dataset import ToothbrushDataset, train_transform
from src.model import get_unet_model
from src.train import train_model
from src.inference import detect_and_flag

def main():
    # --- KONFIGURACJA ---
    # Zmień na True, jeśli chcesz trenować. Zmień na False, jeśli chcesz tylko testować.
    TRAIN_MODE = False
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    TRAIN_DIR = os.path.join(BASE_DIR, 'data', 'train')
    MASK_DIR = os.path.join(BASE_DIR, 'data', 'ground_truth')
    MODEL_WEIGHTS_PATH = os.path.join(BASE_DIR, 'weights', 'best_model.pth')
    # --------------------

    # 1. Inicjalizacja modelu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_unet_model()
    model.to(device)

    # 2. Logika Trenowanie vs Wczytywanie
    if TRAIN_MODE:
        print("\n--- TRYB TRENINGU AKTYWNY ---")
        dataset = ToothbrushDataset(root_dir=TRAIN_DIR, mask_dir=MASK_DIR, transform=train_transform)
        
        if len(dataset) == 0:
            print("Błąd: Brak danych do treningu!")
            return

        train_model(
            model=model, 
            train_dataset=dataset, 
            epochs=25, 
            batch_size=4, 
            save_path=MODEL_WEIGHTS_PATH
        )
        print("Trening zakończony i model zapisany.")
    else:
        print("\n--- TRYB TESTOWANIA (INFERENCE) ---")
        if os.path.exists(MODEL_WEIGHTS_PATH):
            # Wczytujemy zapisane wagi
            model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
            model.eval() # Bardzo ważne: ustawienie modelu w tryb oceny
            print(f"Pomyślnie wczytano model z: {MODEL_WEIGHTS_PATH}")
        else:
            print(f"BŁĄD: Nie znaleziono pliku wag w {MODEL_WEIGHTS_PATH}!")
            print("Musisz najpierw ustawić TRAIN_MODE = True, aby wytrenować model.")
            return

    # # 3. Detekcja na zdjęciu
    # # Tutaj używamy dowolnego zdjęcia, np. z folderu defective
    # test_image_path = os.path.join(TRAIN_DIR, 'good', '019.png') # Upewnij się co do nazwy/rozszerzenia

    # if os.path.exists(test_image_path):
    #     is_defective, mask = detect_and_flag(test_image_path, model)
    #     status = "WADLIWA" if is_defective else "DOBRA"
    #     print(f"\nWynik inspekcji: SZCZOTECZKA {status}")
    # else:
    #     print(f"Nie znaleziono zdjęcia testowego: {test_image_path}")

# 3. Wielowariantowa detekcja (Pętla po folderach good i defective)
    print("\n" + "="*60)
    print(f"{'PROCES TESTOWANIA CAŁEGO ZBIORU':^60}")
    print("="*60)

    # Lista folderów do sprawdzenia
    subfolders = ['good', 'defective']
    TP, TN, FP, FN = 0, 0, 0, 0
    for subfolder in subfolders:
        test_folder = os.path.join(TRAIN_DIR, subfolder)
        
        print(f"\n---> FOLDER: {subfolder.upper()}")
        print(f"{'NAZWA PLIKU':<30} | {'WYNIK ANALIZY':<20}")
        print("-" * 55)

        if os.path.exists(test_folder):
            # Pobieramy listę obrazów
            files = [f for f in os.listdir(test_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            if not files:
                print("   (Brak zdjęć w tym folderze)")
                continue

            # Statystyki dla podfolderu
            correct_count = 0

            for file_name in files:
                full_path = os.path.join(test_folder, file_name)
                
                # Wywołanie Twojej funkcji detekcji
                is_defective, mask = detect_and_flag(full_path, model)
                
                # Formułowanie wyniku i prosta logika oceny poprawności
                if is_defective:
                    status = "🔴 WADLIWA"
                    if subfolder == 'defective': 
                        correct_count += 1
                        TP += 1  # True Positive: Wykryto faktyczną wadę
                    else:
                        FP += 1
                else:
                    status = "🟢 DOBRA"
                    if subfolder == 'good': 
                        correct_count += 1
                        TN += 1  # True Negative: Poprawnie przepuszczono zdrową
                    else:
                        FN += 1
                
                print(f"{file_name:<30} | {status}")

            # Wyświetlenie podsumowania dla folderu
            accuracy = (correct_count / len(files)) * 100
            print("-" * 55)
            print(f"PODSUMOWANIE {subfolder}: Poprawnie rozpoznano {correct_count}/{len(files)} ({accuracy:.1f}%)")
        else:
            print(f"BŁĄD: Folder {test_folder} nie istnieje!")

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    print("\n" + "="*55)
    print(f"{'METRYKI JAKOŚCI':^55}")
    print("="*55)
    print(f"Precision (Precyzja): {precision:.2%}")
    print(f"Recall (Czułość):    {recall:.2%}")
    print(f"F1-Score (Balans):   {f1_score:.2%}")
    print("="*55)

    print("\n" + "="*60)
    print(f"{'KONIEC TESTÓW':^60}")
    print("="*60)

if __name__ == "__main__":
    main()