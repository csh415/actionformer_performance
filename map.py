import pandas as pd
from sklearn.metrics import average_precision_score

def single_iou(gt, start, end):
    intersection = min(gt[2], end) - max(gt[1], start)
    union = max(gt[2], end) - min(gt[1], start)
    iou = intersection / union
    return iou

def extract_labelmatch(df, label):
    df2 = df[df['lab_g'] == label]
    df2.reset_index(drop=True, inplace=True)
    return df2

def extract_overlap(df, start, end, iou_threshold):
    df2 = df.copy()
    for i in range(len(df2)):
        iou = single_iou(df2.iloc[i, :], start, end)
        if iou < iou_threshold:
            df2.loc[i, 'include'] = 'no'
        else:
            df2.loc[i, 'include'] = 'yes'
    df3 = df2[(df2.include == 'yes')].iloc[:, :6]
    df3.reset_index(drop=True, inplace=True)
    return df3

def main(dfp,dfg,iout):
    df=dfp.copy()
    r=range(len(dfp))

    for i in r:
        start = dfp.iloc[i, 1]
        end = dfp.iloc[i, 2]
        label = dfp.iloc[i, 3]
        g0 = dfg[dfg['id'] == (dfp.loc[i, 'id'])]
        g0.reset_index(drop=True, inplace=True)

        for j in range(len(iout)):
            it=iout[j]
            g3 = extract_labelmatch(g0, label)
            if g3.empty:
                result3 = 0
            else:
                g4 = extract_overlap(g3, start, end, it)
                if g4.empty:
                    result3 = 0
                else:
                    result3=1
            df.loc[i, 'Hit'+str(iout[j])] = result3
    return df

def map(df,iout):
    result = pd.DataFrame()
    for i in range(len(iout)):
        y_true = df.iloc[:,5+i].to_numpy()
        y_scores = df[['sco_p']].to_numpy()
        map = average_precision_score(y_true, y_scores)
        result.loc[i,'IoU Threshold']=iout[i]
        result.loc[i, 'mAP'] = map
    return result

iout = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
dfp=pd.read_pickle("dfp.pkl")
dfg=pd.read_pickle("dfg.pkl")
#dfp2=dfp.iloc[:10,:]  #test

df=main(dfp,dfg,iout)
print(map(df,iout))