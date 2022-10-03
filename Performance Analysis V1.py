import json
import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt

def extract_data():
    # Extract Ground Truths Dataframe
    datag = json.load(open('data/thumos/annotations/thumos14.json'))
    dfg_output = pd.DataFrame(datag["database"]).T
    dfg = pd.DataFrame()
    dfg_index = dfg_output.index

    for i in range(len(dfg_output)):
        temp = dfg_output.iloc[i, 3]
        temp_index = dfg_index[i]
        temp_duration = dfg_output.iloc[i, 1]
        temp_fps = dfg_output.iloc[i, 2]
        dfg_temp = pd.DataFrame()

        for j in range(len(temp)):
            dfg_temp.loc[j, 'id'] = temp_index
            dfg_temp.loc[j, 'sta_g'] = temp[j]['segment'][0]
            dfg_temp.loc[j, 'end_g'] = temp[j]['segment'][1]
            dfg_temp.loc[j, 'lab_g'] = temp[j]['label']
            dfg_temp.loc[j, 'fps_g'] = temp_fps
            dfg_temp.loc[j, 'dur_g'] = temp_duration
            dfg_temp.loc[j, 'label_id'] = temp[j]['label_id']
        if i == 0:
            dfg_temp2 = dfg_temp
        else:
            dfg_temp2 = pd.concat([dfg_temp2, dfg_temp])
    dfg_temp2.sort_values(by=['id', 'sta_g'], inplace=True)
    dfg_temp2.reset_index(drop=True, inplace=True)
    dfg = dfg_temp2[["id", "sta_g", "end_g", "lab_g", "fps_g","dur_g"]]

    print(dfg)
    # print('************** Ground Truths dfg pickled ******************')
    dfg.to_pickle('dfg.pkl')

    ### Index to convert label_id to labels
    df1 = dfg_temp2[["lab_g", "label_id"]]
    df2 = df1.drop_duplicates().sort_values(by=["label_id"])
    df2.reset_index(drop=True, inplace=True)

    ### Extract Proposals Dataframe
    data = pd.DataFrame(pd.read_pickle('pretrained/thumos_i3d_reproduce/eval_results.pkl'))
    data.rename(columns={"label": "label_id"}, inplace=True)
    data2 = data.merge(df2, on=["label_id"], how="left")
    data2.rename(columns={"video-id": "id", "t-start": "sta_p", "t-end": "end_p", "score": "sco_p", "lab_g": "lab_p"},
                 inplace=True)
    dfp = data2[["id", "sta_p", "end_p","lab_p","sco_p"]]
    print(dfp)
    # print('************** Proposals dfp pickled ******************')
    dfp.to_pickle('dfp.pkl')
    return


class Analysis:
    def __init__(self,dfg,dfp,record_range,iou_thresholds,verbose):
        self.dfg=dfg
        self.dfp=dfp
        self.record_range=record_range
        self.iou_thresholds=iou_thresholds
        self.verbose=verbose

    def calc_iou(self,df,start,end):
        df2 = df.copy()
        for i in range(len(df2)):
            df2.loc[i, 'inter-s'] = max(start, df2.iloc[i, 1])
            df2.loc[i, 'inter-e'] = min(end, df2.iloc[i, 2])
            df2.loc[i, 'inter-sum'] = df2.loc[i, 'inter-e'] - df2.loc[i, 'inter-s']
            intersection = df2['inter-sum'].sum()
            union = max(end, df.iloc[-1, 2]) - min(start, df.iloc[0, 1])
        iou = intersection / union
        return iou

    def single_iou(self, gt, start, end):  ### calculates iou over a single GT with one proposal
        intersection = min(gt[2], end) - max(gt[1], start)
        union = max(gt[2], end) - min(gt[1], start)
        iou = intersection / union
        return iou

    def extract_labelmatch(self,df, label):
        df2 = df[df['lab_g'] == label]  # temp4: temp3 except labels match proposal
        df2.reset_index(drop=True, inplace=True)
        return df2

    def extract_closest(self,df, start, end):
        if df.empty:
            result2 = 'no relevant GTs'
            delta = 0
            return result2, delta
        else:
            df2 = df.copy()
            for i in range(len(df)):
                if start > df.iloc[i, 2]:
                    df2.loc[i, 'delta'] = start - df.iloc[i, 2]
                    df2.loc[i, 'result'] = 'right miss'
                if end < df.iloc[i, 1]:
                    df2.loc[i, 'delta'] = df.iloc[i, 1] - end
                    df2.loc[i, 'result'] = 'left miss'
            df3 = df2.loc[(df2['result'] == 'right miss') | (df2['result'] == 'left miss')]
            df4 = df3.loc[df3['delta'].idxmin()]
            result2 = df4[7]
            delta = df4[6]
            sta_g=df4[1]
            end_g=df4[2]
            return result2, delta, sta_g, end_g

    def extract_overlap(self,df, start, end, iou_threshold):
        df2 = df.copy()
        for i in range(len(df2)):
            iou = self.single_iou(df2.iloc[i, :], start, end)
            if iou < iou_threshold:
                df2.loc[i, 'include'] = 'no'
            else:
                df2.loc[i, 'include'] = 'yes'
        df3 = df2[(df2.include == 'yes')].iloc[:, :6]
        df3.reset_index(drop=True, inplace=True)
        return df3


    def classify(self, df, start, end):
        df.reset_index(drop=True, inplace=True)
        v_dur = 0
        iou = 0
        ### Result 1, Count of GTs
        result1 = len(df)
        ### Result 2, Classification
        if start < df.iloc[0, 1]:
            if end < df.iloc[-1, 2]:
                result2 = 'left straddle'
            else:
                result2 = 'full straddle'
        else:
            if end < df.iloc[-1, 2]:
                result2 = 'contained'
            else:
                result2 = 'right straddle'

        sta_g=df.iloc[0,1]
        end_g=df.iloc[-1,2]
        ### Result iou
        iou = self.calc_iou(df, start, end)
        return result1, result2, iou, sta_g, end_g


    def main_loop(self):
        df=self.dfp.copy()
        if self.record_range=='all':
            r=range(len(self.dfp))
        else:
            r=range(self.record_range[0],self.record_range[1])

        for i in r:
            ### Setup Variables
            iou = 0
            delta = 0
            result1 = 0
            result2 = ''
            v_dur = 0

            start = self.dfp.iloc[i, 1]
            end = self.dfp.iloc[i, 2]
            label = self.dfp.iloc[i, 3]

            g0 = self.dfg[self.dfg['id'] == (self.dfp.loc[i, 'id'])]
            g0.reset_index(drop=True, inplace=True)
            v_dur = g0.iloc[0, -1]

            ### Show Proposal/GT Data
            if self.verbose:
                print('')
                print(i, '  ---------------')
                print(self.dfp.iloc[i,:5])
                print(g0)

            ### g1 temporal overlap only
            g1 = self.extract_overlap(g0, start, end, iou_threshold=0)
            if len(g1) == 0:
                result2, delta,sta_g,end_g = self.extract_closest(g0, start, end)  #####
            else:
                result1, result2, iou,sta_g,end_g = self.classify(g1, start, end)

            if self.verbose:
                print('')
                print(result1,result2,' iou: ',iou,' delta: ',delta, 'video duration',v_dur,sta_g,end_g)

            df.loc[i, 'Video Duration'] = v_dur
            df.loc[i, 'GT count1'] = result1
            df.loc[i, 'Classification1'] = result2
            df.loc[i, 'IoU1'] = iou
            df.loc[i, 'Delta1'] = delta

            ### g2 temporal overlap and label match
            g2 = self.extract_labelmatch(g1, label)
            if len(g2) == 0:
                g3 = self.extract_labelmatch(g0, label)
                if g3.empty:
                    result1, result2, iou, delta, v_dur, sta_g, end_g = 0, 'no relevant GTs', 0, 0, 0,0,0
                else:
                    result2, delta, sta_g, end_g = self.extract_closest(g3, start, end)
                    result1=0
                    iou=0
            else:
                result1, result2, iou, sta_g,end_g= self.classify(g2, start, end)

            if self.verbose:
                print(result1,result2,' iou: ',iou,' delta: ',delta, 'video duration',v_dur,sta_g,end_g)

            df.loc[i, 'sta_g'] = sta_g
            df.loc[i, 'end_g'] = end_g
            df.loc[i, 'GT count2'] = result1
            df.loc[i, 'Classification2'] = result2
            df.loc[i, 'IoU2'] = iou
            df.loc[i, 'Delta2'] = delta
            df.loc[i, 'Class'] = df.loc[i, 'Classification2'] + ' ' + str(df.loc[i, 'GT count2'])

            if result2 in ['left straddle', 'right straddle', 'full straddle', 'contained']:
                df.loc[i, 'Hit'] = 1
            elif result2 in ['no relevant GTs']:
                df.loc[i, 'Hit'] = -1
            else:
                df.loc[i, 'Hit'] = 0

            ### Calculate lead/lag index
            avg_dist=((df.loc[i,'sta_p']+df.loc[i,'end_p'])/2)-((df.loc[i,'sta_g']+df.loc[i,'end_g'])/2)
            p_length=df.loc[i,'end_g']-df.loc[i,'sta_g']

            if p_length==0:
                df.loc[i, 'Lead/Lag Index'] = 0
            else:
                df.loc[i, 'Lead/Lag Index'] = avg_dist/p_length


            if not self.verbose:
                if self.record_range=='all':
                    print(i, 'out of', len(df))
                else:
                    print(i, 'out of', self.record_range[1])

            ### Mean Average Precision Calc
            # g3 label match only
            for j in range(len(self.iou_thresholds)):
                iout=self.iou_thresholds[j]
                g3 = self.extract_labelmatch(g0, label)
                if g3.empty:
                    result3 = 0
                else:
                    g4 = self.extract_overlap(g3, start, end, iout)
                    if g4.empty:
                        result3 = 0
                    else:
                        result3=1
                        #x,y,iou2=self.classify(g4,start,end)
                df.loc[i, 'Hit'+str(iout)] = result3

                if self.verbose:
                    print('Hit'+str(iout),result3)


        if self.record_range=='all':
            df.to_pickle('dfp2.pkl')
            print('************** dfp2 pickled ******************')
        else:
            print(df.iloc[0,:])
        return

################################################################################################
def split_videosize(df):

    vlmean = df['Video Duration'].mean()
    vlstd = df['Video Duration'].std()

    print('Video length mean: ', vlmean)
    print('Video length std: ', vlstd)

    x1 = vlmean - 0.43 * vlstd
    x2 = vlmean + 0.43 * vlstd

    print('bin 1/2 threshold', x1)
    print('bin 2/3 threshold', x2)

    for i in range(len(df)):
        print(i)
        if df.loc[i, 'Video Duration'] < x1:
            df.loc[i, 'Size'] = 'short'
        elif df.loc[i, 'Video Duration'] > x2:
            df.loc[i, 'Size'] = 'long'
        else:
            df.loc[i, 'Size'] = 'medium'
    df.to_pickle('dfp3.pkl')
    return






def charts(df,title):

    ### Bar chart values
    temp1 = df[['Classification2', "GT count2"]].value_counts(sort=False)
    temp1a = temp1.to_frame().sort_index()

    ### Hits,Misses, no GT values and percentages
    temp2 = df['Hit'].value_counts(sort=False)
    temp2a = temp2.to_frame().sort_index()
    temp2a['percent'] = (temp2a['Hit'] /temp2a['Hit'].sum()) * 100
    nogts = '{:,}'.format(temp2a.iloc[0,0])
    nogts2 = "{:.2%}".format(temp2a.iloc[0, 1]/100)
    misses = '{:,}'.format(temp2a.iloc[1, 0])
    misses2 = "{:.2%}".format(temp2a.iloc[1, 1]/100)
    hits = '{:,}'.format(temp2a.iloc[2, 0])
    hits2 = "{:.2%}".format(temp2a.iloc[2, 1]/100)

    ### Lead / Lag Index
    index = round(df['Lead/Lag Index'].mean(),3)


    ### Values per class
    temp4 = df.groupby(['Classification2']).agg({'Classification2': 'count', 'Lead/Lag Index': 'mean', 'IoU2': 'mean',"Delta2":"mean"})
    temp4.columns = ['Count',"Mean Lead/Lag Index", 'Mean IoU', 'Mean Delta']
    temp4a=temp4.round(2)

    y_val = [temp4a.iloc[2,0], temp4a.iloc[3,0], temp4a.iloc[1,0], temp4a.iloc[0,0], temp4a.iloc[6,0], temp4a.iloc[5,0]]
    lli_val = [temp4a.iloc[2, 1], temp4a.iloc[3, 1], temp4a.iloc[1, 1], temp4a.iloc[0, 1], temp4a.iloc[6, 1],
            temp4a.iloc[5, 1]]
    iou_val = [temp4a.iloc[2, 2], temp4a.iloc[3, 2], temp4a.iloc[1, 2], temp4a.iloc[0, 2], temp4a.iloc[6, 2],
            temp4a.iloc[5, 2]]
    delta_val = [temp4a.iloc[2, 3], temp4a.iloc[3, 3], temp4a.iloc[1, 3], temp4a.iloc[0, 3], temp4a.iloc[6, 3],
              temp4a.iloc[5, 3]]
    y_max=max(y_val)

    ### chart
    ls = temp1a.loc["left straddle"]
    fs = temp1a.loc["full straddle"]
    cx = temp1a.loc["contained"]
    rs = temp1a.loc["right straddle"]
    lx = temp1a.loc["left miss"]
    rx = temp1a.loc["right miss"]

    ls1 =ls.values[0].sum()
    fs1 =fs.values[0].sum()
    c1 =cx.values[0].sum()
    rs1 =rs.values[0].sum()
    lm = lx.values[0].sum()
    rm = rx.values[0].sum()

    if len(ls)>1:
        ls2 = ls.values[1].sum()
    else:
        ls2=0

    if len(fs) > 1:
        fs2 = fs.values[1].sum()
    else:
        fs2=0

    if len(cx) > 1:
        c2 = cx.values[1].sum()
    else:
        c2=0

    if len(rs) > 1:
        rs2 = rs.values[1].sum()
    else:
        rs2=0

    if len(ls) > 2:
        ls3 = ls.values[2:].sum()
    else:
        ls3=0

    if len(fs) > 2:
        fs3 = fs.values[2:].sum()
    else:
        fs3=0

    if len(cx) > 2:
        c3 = cx.values[2:].sum()
    else:
        c3=0

    if len(rs) > 2:
        rs3 = rs.values[2:].sum()
    else:
        rs3=0


    x = ["left miss","left straddle", "full straddle", "contained", "right straddle","right miss"]
    y1 = np.array([0,ls1, fs1, c1, rs1,0])
    y2 = np.array([0,ls2, fs2, c2, rs2,0])
    y3 = np.array([0,ls3, fs3, c3, rs3,0])
    y4=np.array([lm,0,0,0,0,rm])

    plt.bar(x, y1, color="darkgreen")
    plt.bar(x, y2, bottom=y1, color="royalblue")
    plt.bar(x, y3, bottom=y1 + y2, color="indigo")
    plt.bar(x, y4, color="gray")

    ### Bar Chart Value Labels
    for k in range(len(y_val)):
        plt.text(k-0.35,y_val[k]-1*(y_max/25),"Count: "+str(y_val[k]),fontsize=5,color="white")
        plt.text(k - 0.35, y_val[k]-2*(y_max/25), "L/L Index: "+str(lli_val[k]),fontsize=5,color="white")
        plt.text(k - 0.35, y_val[k]-3*(y_max/25), "IoU:" +str(iou_val[k]),fontsize=5,color="white")
        plt.text(k - 0.35, y_val[k]-4*(y_max/25), "Delta: "+str(delta_val[k]),fontsize=5,color="white")

    ### Summary Statistics
    plt.text(3.5, y_max, 'Hits: '+str(hits)+"  "+str(hits2),fontsize=9,color="black")
    plt.text(3.5, y_max-1*(y_max/25), 'Misses: '+str(misses)+"  "+str(misses2), fontsize=9,color="black")
    plt.text(3.5, y_max-2*(y_max/25), 'No Ground Truths: '+str(nogts)+"  "+str(nogts2), fontsize=9,color="black")
    plt.text(3.5, y_max-3*(y_max/25), 'Overall Lead/Lag Index: ' + str(index), fontsize=9,color="black")

    plt.legend(["1", "2", "3+"],title="Ground Truths", loc="upper left",fontsize=8)
    plt.xticks(fontsize=8)
    plt.title("ActionFormer",loc="center")


    fig = plt.gcf()
    plt.show()
    plt.draw()
    fig.savefig("fig_"+title+'.png')
    return


def map(self,iout):
    print('*********************************** mAP')
    result = pd.DataFrame()
    for i in range(len(iout)):
        y_true = self.df.iloc[:,19+i].to_numpy()
        y_scores = self.df[['sco_p']].to_numpy()
        map = average_precision_score(y_true, y_scores)
        result.loc[i,'IoU Threshold']=iout[i]
        result.loc[i, 'mAP'] = map
    print(result)
    result.to_csv("result-map.csv")
    return



def overall_stats(df):
    charts(df, "Overall Statistics")
    return

def by_videosize(df):

    temp1 = df[df["Size"]=="short"]
    temp2 = df[df["Size"] == "medium"]
    temp3 = df[df["Size"] == "long"]

    charts(temp1,"Video Length Short")
    charts(temp2, "Video Length Medium")
    charts(temp3, "Video Length Long")
    return

def by_activity(df):
    act=list(df.lab_p.unique())
    act.sort()

    for i in range(len(act)):  ###????????????
        temp=df[df["lab_p"]==act[i]]
        title="Activity "+str(act[i])
        charts(temp,title)
    return



######################## SUB ROUTINES ###################

def get_data():
    extract_data()
    return

def analyze_data(record_range,iou_thresholds,verbose):
    dfg = pd.read_pickle("dfg.pkl")
    dfp = pd.read_pickle("dfp.pkl")
    temp=Analysis(dfg,dfp,record_range,iou_thresholds,verbose)
    temp.main_loop()

def show_results(iout):
    #df2 = pd.read_pickle("dfp2.pkl")
    #split_videosize(df2)

    df3=pd.read_pickle("dfp3.pkl")
    overall_stats(df3)
    by_videosize(df3)
    by_activity(df3)

    #map(d3f,iout)


############################## MAIN ############################
def main():
    ### Input Parameters
    iout = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    record_range = "all"       #[0,50]
    verbose=False

    #get_data()  ## pickle dfg,dfp
    analyze_data(record_range,iout,verbose)  ## pickle dfp2
    show_results(iout)  ## pickle dfp3 for video split
    return


main()




