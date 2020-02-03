'''List label'''
def getIntLabel():
    label = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9','_']
    return label

def getCharLabel():
    label = ['아', '바', '배', '버', '보', '부', '다', '도', '두', '가',
			 '거', '고', '구', '하', '허', '호', '자', '저', '조', '라',
			 '러', '마', '머', '모', '무', '나', '너', '노', '누', '오',
			 '로', '루', '사', '서', '소', '수', '더', '어', '우', '주']
    return label

def getStrLabel():
    label = ['__','부산', '충북', '충남', '대구', '대전',
             '강원', '광주', '경북', '경남', '인천',
             '제주', '전북', '전남', '경기', '세종',
             '서울', '울산']
    return label

'''Dic label '''
def getIntLabelDic():
    label={}
    for item in getIntLabel():
        label[item] = item
    return label

def getCharLabelDic():
    label = {}
    key = ['ah', 'ba', 'bae','ber', 'bo', 'boo', 'da', 'do', 'du', 'ga',
			'ger', 'go', 'goo', 'ha', 'her', 'ho', 'ja', 'jer', 'jo','la',
			'ler', 'ma', 'mer', 'mo', 'moo', 'na', 'ner', 'no', 'nu', 'oh',
			'ro', 'ru', 'sa', 'ser', 'so', 'su', 'the', 'uh', 'woo','zoo']
    for i,val in enumerate(getCharLabel()):
        label[key[i]] = val
    return label

def getStrLabelDic():
    label = {}
    key = ['__','busan', 'chungbuk', 'chungnam', 'daegu', 'daejeon',
           'gangwon', 'gwangju', 'gyeongbuk', 'gyeongnam', 'incheon',
           'jeju', 'jeonbuk', 'jeonnam', 'kyongi', 'sejong',
           'seoul', 'ulsan']
    for i,val in enumerate(getStrLabel()):
        label[key[i]] = val
    return label


if __name__=='__main__':
    print(getIntLabel())
    print(getCharLabel())
    print(getStrLabel())

    print(getIntLabelDic())
    print(getCharLabelDic())
    print(getStrLabelDic())