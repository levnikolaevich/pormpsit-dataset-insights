import logging
import importlib
from util import stdout_to_err

import warnings as wrns


from sacremoses import MosesTokenizer
from nltk.tokenize import WordPunctTokenizer, word_tokenize


#Apparently mahaNLP overwrites the logging level to quiet-er than desired
logging.disable(logging.NOTSET)

MOSES_LANGS = ["as", "ca", "cs", "co", "de", "el", "en", "es", "fi", "fr", "ga", "hu", "is", "it", "lt", "lv", "mni", "nl", "pl", "pt", "ro", "ru", "sk", "sl", "sv", "ta"]

NLTK_WORD_LANGS = ["ace", "als", "ar", "ayr", "az", "azb", "azj",  "ban", "be", "bem", "bjn", "bm", "bug", "ceb", "cjk", "crh", "dik","dyu", "dz", "ee", "fa", "fj", "fo", "fon", "fur", "fuv",
                    "gaz", "gd", "gn", "ha", "ht", "hy","ilo", "jv", "ka", "kab", "kac", "kam", "kbp", "kea", "kg", "khk", "ki", "kk", "km", "kmb", "knc", "ky", "lg", 
                    "li", "lij", "ln", "lmo",  "ltg", "lua", "luo",  "lus", "mi", "min", "mn", "mos",
                    "ms", "nso", "nus", "ny", "oc", "pag", "pap", "pbt", "pes", "plt", "prs", "ps", "quy", "rn", "rw", "sc", "scn", "sd", "sg", "sm", "sn", "ss", "st", "su", "swh", "szl",
                    "taq", "tg", "tk", "tn", "tpi", "ts", "tt", "tw", "tzm", "tum", "uk", "ug", "umb", "vec", "vi", "war", "wo", "xh", "yo","zsm",  "zu"]
NLTK_PUNKT_LANGS = {"no": "norwegian",
                    "et": "estonian",
                    "da": "danish",
                    "tr": "turkish"}

RELDI_LANGS  = ["sr", "mk", "bg", "hr"]
RELDI_FALLBACK = {"bs": "sr"}

MOSES_FALLBACK = {
                 "ab": "ru",
                 "af": "nl",
                 "ast": "es",
                 "ba": "ru",
                 "br": "fr",
                 "co": "it",
                 "cy": "en",
                 "eo": "en",
                 "eu": "es",
                 "gl": "es",
                 "la": "en",
                 "lb": "de",
                 "lvs": "lv",
                 "mt": "en",
                 "so": "en",
                 "sq": "en",
                 "sw": "en",
                 "tl": "en"}

NLTK_FALLBACK = {"nb": "no",
                "nn": "no"}

MECAB_JA = ["ja"]

MECAB_KO = ["ko"]

PDS_LANGS = ["my", "shn"]

SINLING_LANGS = ["si"]

FITRAT_LANGS = ["uz", "uzn"]

BNLP_LANGS = ["bn"]

THAI_LANGS = ["th"]

INDIC_LANGS = ["awa", "bho", "gu" ,"hi", "hne", "kn", "ks", "mag", "mai", "ml", "mr", "ne", "npi", "pa", "sa", "sat", "te", "ur"]

PKUSEG_LANGS = ["zh", "zh-Hant"]

HEBREW_LANGS = ["he", "iw", "ydd"]

NLPID_LANGS =  ["id"]

#BOTOK_LANGS = ["bo"]	#Removed because of poor performance, temporarily using NLTK
BOTOK_LANGS=[]
NLTK_WORD_LANGS.append("bo")

#KLPT_LANGS = ["ckb", "kmr"]  # Removed because of poor performance, temporarily using NLTK
KLPT_LANGS=[]
NLTK_WORD_LANGS.append("ckb")
NLTK_WORD_LANGS.append("kmr")


CANTONESE_LANGS = ["yue"]

LAONLP_LANGS = ["lo"]

OPENODIA_LANGS = ["ory"]

IGBO_LANGS = ["ig"]

ETHIOPIC_LANGS = ["am", "ti"]

class CustomTokenizer:


    def  __init__(self, lang):
        self.lang = None
        self.tokenizer = None
        self.toktype = None
        self.warnings = []
        
        self.lang = lang
        self.setTokenizer(lang)

    def _fallback_missing(self, module_name):
        self.tokenizer = WordPunctTokenizer()
        self.toktype = "nltk_wordpunct"
        self.warnings.append(f"warning_tok_missing_{module_name}")
        self.warnings.append("warning_tok_nltk_wordpunct")
    

    def setTokenizer(self, lang):
        if lang in MOSES_LANGS:
            self.tokenizer =  MosesTokenizer(lang).tokenize
            self.toktype = "moses"
        elif lang in MOSES_FALLBACK.keys():
            moses_lang = MOSES_FALLBACK.get(lang)
            self.tokenizer = MosesTokenizer(moses_lang).tokenize
            self.toktype =  "moses"
            self.warnings.append("warning_tok_moses_"+moses_lang)

        elif lang in NLTK_WORD_LANGS:
            self.tokenizer = WordPunctTokenizer()
            self.toktype = "nltk_wordpunct"
            self.warnings.append("warning_tok_nltk_wordpunct")
        elif lang in NLTK_PUNKT_LANGS.keys():
            self.tokenizer = word_tokenize
            self.toktype = "nltk_punkt_" + NLTK_PUNKT_LANGS.get(lang)
        elif lang in NLTK_FALLBACK.keys():
            nltk_langcode = NLTK_FALLBACK.get(lang)
            nltk_langname = NLTK_PUNKT_LANGS.get(nltk_langcode)
            self.tokenizer = word_tokenize
            self.toktype = "nltk_punkt_" + nltk_langname
            self.warnings.append("warning_tok_nltk_punkt_"+nltk_langcode)           

        elif lang in MECAB_JA:
            try:
                MeCab = importlib.import_module("MeCab")
                self.tokenizer = MeCab.Tagger("-Owakati")
                self.toktype = "mecab"
            except ImportError:
                self._fallback_missing("MeCab")

        elif lang in MECAB_KO:
            try:
                mecab_ko = importlib.import_module("mecab_ko")
                self.tokenizer = mecab_ko.Tagger("-Owakati")
                self.toktype = "mecab"
            except ImportError:
                self._fallback_missing("mecab_ko")

#        elif lang in NLPASHTO_LANGS:
#            self.tokenizer = word_segment(sent)
#            self.toktype = "nlpashto"
        
        elif lang in RELDI_LANGS:
            try:
                reldi_tokeniser = importlib.import_module("reldi_tokeniser")
                self.tokenizer = reldi_tokeniser
                self.toktype = "reldi_" + lang
            except ImportError:
                self._fallback_missing("reldi_tokeniser")
        elif lang in RELDI_FALLBACK.keys():
            try:
                reldi_tokeniser = importlib.import_module("reldi_tokeniser")
                self.tokenizer = reldi_tokeniser
                reldilang = RELDI_FALLBACK.get(lang)
                self.toktype = "reldi_" + reldilang
                self.warnings.append("warning_tok_reldi_"+reldilang)
            except ImportError:
                self._fallback_missing("reldi_tokeniser")
        
        elif lang in PDS_LANGS:
            try:
                pyidaungsu = importlib.import_module("pyidaungsu")
                self.tokenizer = pyidaungsu.tokenize
                self.toktype = "pds"
            except ImportError:
                self._fallback_missing("pyidaungsu")
            
        elif lang in SINLING_LANGS:
            try:
                SinhalaTokenizer = importlib.import_module("sinling").SinhalaTokenizer
                self.tokenizer = SinhalaTokenizer()
                self.toktype = "sinling"
            except ImportError:
                self._fallback_missing("sinling")

        elif lang in FITRAT_LANGS:
            try:
                fitrat_word_tokenize = importlib.import_module("fitrat").word_tokenize
                self.tokenizer = fitrat_word_tokenize
                self.toktype = "fitrat"
            except ImportError:
                self._fallback_missing("fitrat")
            
        elif lang in BNLP_LANGS:
            try:
                NLTKTokenizer = importlib.import_module("bnlp").NLTKTokenizer
                self.tokenizer = NLTKTokenizer()
                self.toktype = "bnlp"
            except ImportError:
                self._fallback_missing("bnlp")

        elif lang in THAI_LANGS:
            try:
                thai_tokenize = importlib.import_module("thai_segmenter").tokenize
                self.tokenizer = thai_tokenize
                self.toktype = "thai"
            except ImportError:
                self._fallback_missing("thai_segmenter")

        elif lang in INDIC_LANGS:
            try:
                indic_tokenize = importlib.import_module("indicnlp.tokenize.indic_tokenize")
                self.tokenizer = indic_tokenize
                self.toktype = "indic_" + lang
            except ImportError:
                self._fallback_missing("indicnlp")
           
        elif lang in PKUSEG_LANGS:
            try:
                pkuseg = importlib.import_module("spacy_pkuseg")
                self.tokenizer = pkuseg.pkuseg()
                self.toktype = "pkuseg"
            except ImportError:
                self._fallback_missing("spacy_pkuseg")
                
        elif lang in HEBREW_LANGS:
            try:
                with wrns.catch_warnings(), stdout_to_err():
                    wrns.simplefilter(action='ignore', category=FutureWarning)
                    hebrew_tokenizer = importlib.import_module("hebrew_tokenizer")
                self.tokenizer = hebrew_tokenizer
                self.toktype = "hebrew"
            except ImportError:
                self._fallback_missing("hebrew_tokenizer")
            
        elif lang in NLPID_LANGS:
            try:
                IndonesianTokenizer = importlib.import_module("nlp_id.tokenizer").Tokenizer
                self.tokenizer = IndonesianTokenizer()
                self.toktype = "nlpid"
            except ImportError:
                self._fallback_missing("nlp_id")

        elif lang in BOTOK_LANGS:  #This is a little bit slow...
            try:
                botok = importlib.import_module("botok")
                config = botok.config.Config(dialect_name="general")
                self.tokenizer = botok.WordTokenizer(config)
                self.toktype = "botok"
            except ImportError:
                self._fallback_missing("botok")


        elif lang in KLPT_LANGS:
            try:
                KurdishTokenizer = importlib.import_module("klpt.tokenize").Tokenize
                if lang == "kmr":
                    self.tokenizer = KurdishTokenizer("Kurmanji", "Latin")
                elif lang == "ckb":
                    self.tokenizer = KurdishTokenizer("Sorani", "Arabic")
                self.toktype = "klpt"
            except ImportError:
                self._fallback_missing("klpt")
        
        elif lang in CANTONESE_LANGS:
            try:
                cantonese_segment = importlib.import_module("pycantonese.word_segmentation").segment
                self.tokenizer = cantonese_segment
                self.toktype = "pycantonese"
            except ImportError:
                self._fallback_missing("pycantonese")
        
        elif lang in LAONLP_LANGS:
            try:
                lao_tokenize = importlib.import_module("laonlp.tokenize").word_tokenize
                self.tokenizer = lao_tokenize
                self.toktype = "laonlp"
            except ImportError:
                self._fallback_missing("laonlp")
            
        elif lang in OPENODIA_LANGS:
            try:
                openodia_tokenize = importlib.import_module("openodia.ud").word_tokenizer
                self.tokenizer = openodia_tokenize
                self.toktype = "openodia"
            except ImportError:
                self._fallback_missing("openodia")
            
        elif lang in IGBO_LANGS:
            try:
                IgboText = importlib.import_module("igbo_text").IgboText
                self.tokenizer = IgboText()
                self.toktype = "igbo"
            except ImportError:
                self._fallback_missing("igbo_text")
        
        elif lang in ETHIOPIC_LANGS:
            try:
                if lang == "ti":
                    tigrinya_tokenizer = importlib.import_module("etnltk.tokenize.tg").word_tokenize
                    self.tokenizer = tigrinya_tokenizer
                elif lang == "am":
                    amharic_tokenizer = importlib.import_module("etnltk.tokenize.am").word_tokenize
                    self.tokenizer = amharic_tokenizer
                self.toktype = "ethiopic"
            except ImportError:
                self._fallback_missing("etnltk")
            
        else:
            '''
            self.tokenizer =  MosesTokenizer("en")
            self.toktype = "moses"
            self.warnings.append("warning_tok_moses_en")
            '''
            self.tokenizer = WordPunctTokenizer()
            self.toktype = "nltk_wordpunct"
            self.warnings.append("warning_tok_nltk_wordpunct")

    def tokenize(self, sent):
        try:    
            if self.toktype == "moses":
                return self.tokenizer(sent, escape=False)
            
            elif self.toktype == "nltk_wordpunct" :
                return self.tokenizer.tokenize(sent)
   
            elif self.toktype.startswith("nltk_punkt_"):
                nltk_lang = self.toktype.split("_")[2]
                return self.tokenizer(sent, language=nltk_lang)
   
            elif self.toktype == "mecab":
                return self.tokenizer.parse(sent).split()
   
            elif self.toktype.startswith("reldi_"):
                #tokstring looks like "'1.1.1.1-5\tHello\n1.1.2.6-6\t,\n1.1.3.8-11\tgood\n1.1.4.13-19\tmorning\n\n'"
                reldi_lang = self.toktype.split("_")[1]
                tokstring =  self.tokenizer.run(lang=reldi_lang, text=sent)
                tokens = []
                for token in tokstring.split("\t"):
                    if "\n" in  token:
                        tokens.append(token.split("\n")[0])
                return tokens        
   
            elif self.toktype == "pds":
                return self.tokenizer(sent, lang="mm", form="word")


            elif self.toktype == "sinling":
                return self.tokenizer.tokenize(sent)

            elif self.toktype == "fitrat":
                return self.tokenizer(sent)
        

            elif self.toktype == "bnlp":
                return self.tokenizer.word_tokenize(text=sent)
        
            elif self.toktype == "thai": 
                return [t for t in self.tokenizer(sent) if t != " "] #This tokenizer returns empty spaces too
     
            elif self.toktype.startswith("indic_"):
                indic_lang = self.toktype.split("_")[1]
                return self.tokenizer.trivial_tokenize(sent, lang=indic_lang)
        
            elif self.toktype == "pkuseg":
                return self.tokenizer.cut(sent)
        
            elif self.toktype == "hebrew":            
                objs = self.tokenizer.tokenize(sent) #this is a generator of objects: ('HEBREW', 'למכולת', 9, (41, 47))  (The hebrew word is in index 1, but RTL languages messing it all)
                tokens = []
                for obj in objs:
                    tokens.append(obj[1])
                return tokens
            
            elif self.toktype == "nlpid":
                return self.tokenizer.tokenize(sent)    
        
            elif self.toktype == "botok":
                objects = self.tokenizer.tokenize(sent)
                tokens = []
                for obj in objects:
                    tokens.append(obj.text)
                return tokens
                
            elif self.toktype == "klpt":
                tokens = self.tokenizer.word_tokenize(sent, keep_form=True, separator= " ")
                return tokens
                
            elif self.toktype == "pycantonese":
                tokens = self.tokenizer(sent)
                return tokens
                
            elif self.toktype == "laonlp":
                tokens = self.tokenizer(sent)
                return tokens
            
            elif self.toktype == "openodia":
                tokens = self.tokenizer(sent)
                return tokens            
            
            elif self.toktype == "igbo":
                tokens = self.tokenizer.tokenize(sent)
                return tokens
                
            elif self.toktype == "ethiopic":
                tokens = self.tokenizer(sent, return_word=False)
                return tokens
                
            else:
                return None #TO DO Do something better here --> Because THIS CRASHES
        except Exception as ex:
            logging.error("Failed at tokenizing: " + sent)
            logging.error(ex)
            return []
    

    def getWarnings(self):
        return self.warnings
