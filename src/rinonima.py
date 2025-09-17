import os 

cartella = 'Data/Edf/'

file_edf = [f for f in os.listdir(cartella) if f.endswith('.edf')]

for i, filename in enumerate(file_edf, 1):
        
        vecchio_nome = os.path.join(cartella, filename)
        nuovo_nome = os.path.join(cartella, f"{i}.edf")
        
        # Rinominare il file
        os.rename(vecchio_nome, nuovo_nome)
        print(f"Rinominato: {vecchio_nome} -> {nuovo_nome}")