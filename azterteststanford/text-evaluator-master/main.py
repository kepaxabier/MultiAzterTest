from file_loader import FileLoader
from printer import Printer
import nltk
####Argumentos##################################
from argparse import ArgumentParser
####analizador sintactico#######################
import stanfordnlp
import os

def start():
    from analyzer import Analyzer
    p = ArgumentParser(description="python3 ./main.py -f \"laginak/*.doc.txt\" ")
    optional = p._action_groups.pop()  # Edited this line
    required = p.add_argument_group('Required arguments')
    required.add_argument("-f", "--files", nargs='+', help="Files to analyze (in .txt, .odt, .doc or .docx format)")
    optional.add_argument('-a', '--all', action='store_true', help="Generate a CSV file with all the results")
    optional.add_argument('-s', '--similarity', action='store_true', help="Calculate similarity (max. 5 files)")
    p._action_groups.append(optional)
    opts = p.parse_args()
    FileLoader.load_files(opts.files)
    FileLoader.load_irregular_verbs_list()
    FileLoader.load_dale_chall_list()
    FileLoader.load_connectives_list()
    FileLoader.load_oxford_word_list()
    #Cargar el analizador
    #MODELS_DIR = '/home/kepa/stanfordnlp_resources/en_ewt_models'
    #stanfordnlp.download('en', MODELS_DIR)
    stanford = stanfordnlp.Pipeline() 
    df_row = None
    ### Files will be created in this folder
    path = Printer.create_directory(FileLoader.files[0])
    file_num = 0
    total = len(FileLoader.files)
    for input in FileLoader.files:
        texto = Analyzer.process_text(input=input)
        # Analizar
        a = Analyzer(texto, input, stanford)
        i = a.analyze(opts.similarity)
        df = a.create_dataframe()
        #prediction = a.predict_dificulty(df)
        prediction="kk"
        file_num += 1
        p = Printer(input, i)
        p.print_info(opts.similarity, prediction, file_num, total)
        if opts.all:
            df_row = p.write_in_full_csv(df_row, opts.similarity)
        p.generate_csv(path, prediction, opts.similarity)
    if opts.all:
        df_row.to_csv(os.path.join(path, "full_results_aztertest.csv"), encoding='utf-8', index=False)

nltk.download('cmudict')
nltk.download('punkt')  # ,download_dir='/home/lsi/metrix-env/nltk_data')
nltk.download('stopwords')
nltk.download('wordnet')
start()
