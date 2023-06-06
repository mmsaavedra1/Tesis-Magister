import pickle
import pandas as pd


def pickle_to_excel(filename: str, case: int, model_name: str):
  """
  Función que transforma un pickle de variables en un excel de matrices.
  
  Input:
  - filename: Nombre del archivo de input.
  - case: Número de identificación del caso de estudio.
  """  
  objects = []
  for letra in ['D', 'L', 'P', 'S', 'X']:
      nombre = f"{model_name}/Output/Variables/{case}/{filename}_{letra}"
      with (open(f"{nombre}.pkl", "rb")) as openfile:
          objects.append((f"{filename}_{letra}", pickle.load(openfile)))

  excel = pd.ExcelWriter(f'{model_name}/Output/Excel/output_variables.xlsx')

  for nombre, df in objects:
      df.to_excel(excel, nombre)

  excel.save()

