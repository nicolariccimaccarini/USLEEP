# Tirocinio
# Riconoscimento degli Spindle con Apprendimento Non Supervisionato

## Descrizione del Progetto  
Questo progetto è stato sviluppato durante il mio tirocinio e si concentra sull'uso di tecniche di **apprendimento non supervisionato** per il **riconoscimento degli spindle** nelle registrazioni di segnali biologici.  
Gli spindle sono eventi oscillatori tipici del sonno e la loro identificazione automatica è fondamentale per lo studio del sonno e dei disturbi neurologici.

## Obiettivi  
- Applicare **metodi di clustering** per il riconoscimento automatico degli spindle.  
- Analizzare le caratteristiche temporali e frequenziali dei segnali.  
- Confrontare diversi algoritmi non supervisionati per valutare le loro prestazioni.  

## Metodologia  
1. **Preprocessing dei dati**:
   - Segmentazione
   - Normalizzazione dei segnali  
   - Calcolo dello spettogramma  
3. **Apprendimento Non Supervisionato**:
   - Estrazione delle feature tramite autoencoder 
   - Applicazione di algoritmi **K-Means** 
   - Analisi delle distribuzioni delle feature  
5. **Valutazione e Interpretazione**:  
   - Validazione dei cluster ottenuti  
   - Confronto con annotazioni 

## Strumenti e Tecnologie  
- **Linguaggi**: Python  
- **Librerie**: NumPy, Pandas, Tensorflow, MNE  
- **Dataset**: Registrazioni EEG contenenti eventi spindle  

## Risultati Attesi  
- Identificazione degli spindle con buona accuratezza. 
- Potenziale utilizzo del modello in studi clinici e ricerca sul sonno.  
