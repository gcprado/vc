import csv
import re

# Input and output CSV file paths
input_csv = "out/reporte_ocr.csv"
output_csv = "out/matriculas_filtradas.csv"

# Regex para OCR_Paddle: 4 dígitos seguidos de 3 letras (case-insensitive)
pattern = re.compile(r"^\d{4}[A-Za-z]{3}$")

def frame_to_time(frame):
    """Convierte número de frame a formato M:SS."""
    total_seconds = frame / 25
    minutes = int(total_seconds // 60)
    seconds = int(total_seconds % 60)
    return f"{minutes}:{seconds:02d}"

# Diccionario para guardar la mejor fila por matrícula
best_rows = {}

with open(input_csv, newline='', encoding='utf-8') as infile:
    reader = csv.DictReader(infile)
    fieldnames = reader.fieldnames.copy()

    # Reemplazamos "frame" por "time" en los encabezados
    if "frame" in fieldnames:
        fieldnames[fieldnames.index("frame")] = "time"

    for row in reader:
        try:
            conf_paddle = float(row["Conf_Paddle"])
            ocr_paddle = row["OCR_Paddle"].strip()
            frame = int(row["frame"])
        except (KeyError, ValueError):
            continue  # salta filas con datos inválidos
        
        if conf_paddle > 0.60 and pattern.match(ocr_paddle):
            row["time"] = frame_to_time(frame)
            del row["frame"]

            # Si ya hay una fila para esta matrícula, guarda la de mayor confianza
            if ocr_paddle not in best_rows or conf_paddle > float(best_rows[ocr_paddle]["Conf_Paddle"]):
                best_rows[ocr_paddle] = row

# Escribir las mejores filas al nuevo CSV
with open(output_csv, 'w', newline='', encoding='utf-8') as outfile:
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(best_rows.values())

print(f"✅ Filtrado completo. Resultados guardados en '{output_csv}'.")
