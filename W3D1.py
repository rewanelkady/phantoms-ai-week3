import numpy as np
import pandas as pd
import seaborn as sns


arr = np.array([[1,2],[3,4]])
print (arr)
print (np.mean(arr))
print (np.std(arr))

df = sns.load_dataset('iris')
print(df.head(5))
print(df.columns)



# Dataset (Iris) تعتبر مناسبة لمسائل الـ Supervised Learning .
# لأن البيانات تكون مصنفة ومعلمة مسبقاً،حيث يحتوي الجدول على عمود الإجابة 
# مما يتيح للنموذج التعلم من هذه البيانات المصنفة للتنبؤ بأنواع جديدة.