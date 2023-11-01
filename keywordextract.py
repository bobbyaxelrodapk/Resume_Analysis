import fitz
import string
import spacy
import gensim
import pandas as pd

nlp=spacy.load('en_core_web_sm')


def extract_key_word(text):
    doc=nlp(text)
    text_words=[]
    for token in doc:
        if token.is_stop==False and token.is_punct==False and token.like_num==False and token.is_bracket==False and token.text not in ['\n','/','|','•',' ']:
            text_words.append(token.lemma_)
            
    terms = {'Operation management':['automation','bottleneck','cycle time','efficiency','fmea',
                                   'machinery','maintenance','manufacture','line balancing','oee','operations',
                                   'operations research','optimization','overall equipment effectiveness',
                                   'pfmea','process mapping','production','value stream mapping',
                                   'utilization','black belt','capability analysis','control charts','doe','dmaic',
                                   'fishbone','gage r&r','green belt','ishikawa','iso','kaizen','kpi','lean','metrics',
                                   'pdsa','performance improvement','process improvement','quality','quality circles',
                                   'quality tools','root cause','six sigma','abc analysis','eoq','epq','inventory','logistic',
                                   'third party logistics','supply chain','feasibility analysis','kanban','pmi','pmp'],
        'Data analytics':['analytics','api','aws','big data','busines intelligence','clustering','code',
                          'coding','data','database','data mining','data science','deep learning','hadoop',
                          'hypothesis test','iot','internet','machine learning','modeling','nosql','nlp',
                          'predictive','programming','python','r','sql','tableau','text mining',
                          'visualization'],
        'Finance':['finance','financial analysis','financial reporting','business planning','corporate finance',
                   'business strategy','accounting','budgeting','mergers & acquisitions','forecasting','Change Management',
                   'management consulting','compliance','valuation','risk assesment','return on investment',
                   'cost-benefit analysis','strategic financial planning','asset management','Profit','P&L Management'],
        'Human resources':['compensation','benefits','diversity and inclusion','employee communications',
                            'employee relations','human resource','information systems','job analysis','labor','negotiations',
                            'leadership development','manpower planning','onboarding','organizational development',
                            'performance management','recruitment','retention','staffing','succession planning',
                            'talent development','Training and development'],
         'Marketing':['marketing','marketing strategy','digital marketing','social media marketing','online marketing',
                      'marketing management','social media','product marketing','event management','advertising',
                      'project management','business strategy','brand management','market research','strategic marketing',
                      'distribution channels','product launch','public relations','pricing strategies']}        
    
    ext_list = {}
    for area in terms.keys():
        ext_list[area]=[]
        for word in terms[area]:
            if word in text_words:
                ext_list[area].append(word)
                
    ext_list = {k:v for k, v in ext_list.items() if v}
    ext_list = {k:[len(v), v] for k, v in ext_list.items()}
    
    return ext_list