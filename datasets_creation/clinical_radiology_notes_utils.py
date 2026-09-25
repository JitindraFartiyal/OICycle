import pandas as pd


def remove_word(ser: pd.Series, word: str) -> pd.Series:
    return ser.str.replace(pat=word, repl='')


def clean(ser: pd.Series) -> pd.Series:
    ser = ser.str.replace(pat=r'\n', repl='', regex=True)
    ser = ser.str.replace(pat=r'\s+', repl=' ', regex=True)

    return ser

notes_regex ={
    'Allergies': {
    'pat': r'Allergies:\s*(\S.*?)(?=\s*Attending)',
    'rm_word': ['Allergies:', 'Attending'],
    },

    'Chief Complaint': {
        'pat': r'Chief Complaint:\s*(\S.*?)(?=\s*Major Surgical or Invasive Procedure:)',
        'rm_word': ['Chief Complaint:', 'Major Surgical or Invasive Procedure:'],
    },

}

radiology_regex ={
    'Findings':{
        'rm_word': ['\n']
    }
}