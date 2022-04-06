"""
Funcoes auxiliares para trabalhar no projeto.
"""

import kaggle
import shutil
import os
import traceback
import logging

COMPETITION_FOLDER = 'jpx'
COMPETITION_NAME = 'jpx-tokyo-stock-exchange-prediction'

def download_competition_files(
    competition_folder: str = COMPETITION_FOLDER,
    competition_name: str = COMPETITION_NAME,
) -> bool:
    """
    Faz o download de todos os arquivos relacionados a competição
    e os extrai na pasta indicada (competition_folder).
    """

    try:

        logging.info('Baixando arquivos do Kaggle.')
        _ = kaggle.api.competition_download_files(competition=competition_name)

        logging.info(f'Criando a pasta {competition_name} caso não exista.')
        _ = os.makedirs(competition_folder, exist_ok=True)

        logging.info(f'Extraindo arquivo zip na pasta definida no script.')
        _ = shutil.unpack_archive(f'./{competition_name}.zip', extract_dir=competition_folder)

        logging.info(f'Deletando o arquivo ZIP final.')
        _ = os.remove(f'./{competition_name}.zip')

    except:
        
        print(traceback.format_exc())

        raise
