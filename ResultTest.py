from kaggle import getPredictions
from kaggle import encode_sentences


input=encode_sentences(['Quality Profile Access'])
result=getPredictions('finalized_model.sav', input)

print("Intent Name received is : "+str(result))

