import pandas as pd
import networkx as nx
import numpy as np
import gurobipy
import warnings

# 分页显示数据, 设置为 False 不允许分页
pd.set_option('display.expand_frame_repr', False)
# 最多显示的列数, 设置为 None 显示全部列
pd.set_option('display.max_columns', None)
# 最多显示的行数, 设置为 None 显示全部行
pd.set_option('display.max_rows', None)

class DEA(object):
    def __init__(self, DMUs_Name, X, Y, AP=False):
        self.m1, self.m1_name, self.m2, self.m2_name, self.AP = X.shape[1], X.columns.tolist(), Y.shape[
            1], Y.columns.tolist(), AP
        self.DMUs, self.X, self.Y = gurobipy.multidict(
            {DMU: [X.loc[DMU].tolist(), Y.loc[DMU].tolist()] for DMU in DMUs_Name})
        print(f'DEA(AP={AP}) MODEL RUNNING...')

        # 检查输入数据
        for DMU in self.DMUs:
            print(DMU)
            if any(np.isnan(self.X[DMU])) or any(np.isnan(self.Y[DMU])) or any(np.isinf(self.X[DMU])) or any(np.isinf(self.Y[DMU])):
                raise ValueError(f"数据问题在 DMU: {DMU}，X 或 Y 包含 NaN 或 Inf 值")


    def __CCR(self):
        for k in self.DMUs:
            MODEL = gurobipy.Model()
            OE, lambdas, s_negitive, s_positive = MODEL.addVar(), MODEL.addVars(self.DMUs), MODEL.addVars(
                self.m1), MODEL.addVars(self.m2)
            MODEL.update()
            MODEL.setObjectiveN(OE, index=0, priority=1)
            MODEL.setObjectiveN(-(sum(s_negitive) + sum(s_positive)), index=1, priority=0)
            MODEL.addConstrs(
                gurobipy.quicksum(lambdas[i] * self.X[i][j] for i in self.DMUs if i != k or not self.AP) + s_negitive[
                    j] == OE * self.X[k][j] for j in range(self.m1))
            MODEL.addConstrs(
                gurobipy.quicksum(lambdas[i] * self.Y[i][j] for i in self.DMUs if i != k or not self.AP) - s_positive[
                    j] == self.Y[k][j] for j in range(self.m2))
            MODEL.setParam('OutputFlag', 0)
            MODEL.optimize()

            if MODEL.status == gurobipy.GRB.Status.OPTIMAL:
                self.Result.at[k, ('效益分析', '综合技术效益(CCR)')] = MODEL.objVal
                self.Result.at[k, ('规模报酬分析',
                                   '有效性')] = '非 DEA 有效' if MODEL.objVal < 1 else 'DEA 弱有效' if s_negitive.sum().getValue() + s_positive.sum().getValue() else 'DEA 强有效'
                self.Result.at[k, ('规模报酬分析',
                                   '类型')] = '规模报酬固定' if lambdas.sum().getValue() == 1 else '规模报酬递增' if lambdas.sum().getValue() < 1 else '规模报酬递减'
                for m in range(self.m1):
                    self.Result.at[k, ('差额变数分析', f'{self.m1_name[m]}')] = s_negitive[m].X
                    self.Result.at[k, ('投入冗余率', f'{self.m1_name[m]}')] = 'N/A' if self.X[k][m] == 0 else s_negitive[
                                                                                                             m].X / \
                                                                                                         self.X[k][m]
                for m in range(self.m2):
                    self.Result.at[k, ('差额变数分析', f'{self.m2_name[m]}')] = s_positive[m].X
                    self.Result.at[k, ('产出不足率', f'{self.m2_name[m]}')] = 'N/A' if self.Y[k][m] == 0 else s_positive[
                                                                                                             m].X / \
                                                                                                         self.Y[k][m]
            else:
                self.Result.at[k, ('效益分析', '综合技术效益(CCR)')] = 'N/A'
                self.Result.at[k, ('规模报酬分析', '有效性')] = 'N/A'
                self.Result.at[k, ('规模报酬分析', '类型')] = 'N/A'
                for m in range(self.m1):
                    self.Result.at[k, ('差额变数分析', f'{self.m1_name[m]}')] = 'N/A'
                    self.Result.at[k, ('投入冗余率', f'{self.m1_name[m]}')] = 'N/A'
                for m in range(self.m2):
                    self.Result.at[k, ('差额变数分析', f'{self.m2_name[m]}')] = 'N/A'
                    self.Result.at[k, ('产出不足率', f'{self.m2_name[m]}')] = 'N/A'
        return self.Result

    def __BCC(self):
        for k in self.DMUs:
            MODEL = gurobipy.Model()
            TE, lambdas = MODEL.addVar(), MODEL.addVars(self.DMUs)
            MODEL.update()
            MODEL.setObjective(TE, sense=gurobipy.GRB.MINIMIZE)
            MODEL.addConstrs(
                gurobipy.quicksum(lambdas[i] * self.X[i][j] for i in self.DMUs if i != k or not self.AP) <= TE *
                self.X[k][j] for j in range(self.m1))
            MODEL.addConstrs(
                gurobipy.quicksum(lambdas[i] * self.Y[i][j] for i in self.DMUs if i != k or not self.AP) >= self.Y[k][j]
                for j in range(self.m2))
            MODEL.addConstr(gurobipy.quicksum(lambdas[i] for i in self.DMUs if i != k or not self.AP) == 1)
            MODEL.setParam('OutputFlag', 0)
            MODEL.optimize()

            if MODEL.status == gurobipy.GRB.Status.OPTIMAL:
                self.Result.at[k, ('效益分析', '技术效益(BCC)')] = MODEL.objVal
            else:
                self.Result.at[k, ('效益分析', '技术效益(BCC)')] = 'N/A'
        return self.Result

    def dea(self):
        columns_Page = ['效益分析'] * 3 + ['规模报酬分析'] * 2 + ['差额变数分析'] * (self.m1 + self.m2) + ['投入冗余率'] * self.m1 + [
            '产出不足率'] * self.m2
        columns_Group = ['技术效益(BCC)', '规模效益(CCR/BCC)', '综合技术效益(CCR)', '有效性', '类型'] + (self.m1_name + self.m2_name) * 2
        self.Result = pd.DataFrame(index=self.DMUs, columns=[columns_Page, columns_Group])
        self.__CCR()
        self.__BCC()

        # 在计算之前确认所有数据都是数值类型，将非数值转换为 NaN
        self.Result.loc[:, ('效益分析', '综合技术效益(CCR)')] = pd.to_numeric(self.Result.loc[:, ('效益分析', '综合技术效益(CCR)')],
                                                                    errors='coerce')
        self.Result.loc[:, ('效益分析', '技术效益(BCC)')] = pd.to_numeric(self.Result.loc[:, ('效益分析', '技术效益(BCC)')],
                                                                  errors='coerce')

        CCR_efficiency = self.Result.loc[:, ('效益分析', '综合技术效益(CCR)')]
        BCC_efficiency = self.Result.loc[:, ('效益分析', '技术效益(BCC)')]
        BCC_efficiency_replaced = BCC_efficiency.replace(0, np.nan)  # 将零值替换为 NaN

        self.Result.loc[:, ('效益分析', '规模效益(CCR/BCC)')] = CCR_efficiency / BCC_efficiency_replaced

        return self.Result

    def analysis(self, file_name=None):
        Result = self.dea()
        file_name = r'DEA 数据包络分析报告.xlsx'
        Result.to_excel(file_name, 'DEA 数据包络分析报告')


# 求解逆序数
class InverseOrder:
    def __init__(self, arr):
        self.arr = arr

    def merge_and_count(self, left, mid, right):
        arr = self.arr
        left_subarray = arr[left:mid + 1]
        right_subarray = arr[mid + 1:right + 1]

        inversions = 0
        i = j = 0
        k = left

        while i < len(left_subarray) and j < len(right_subarray):
            if left_subarray[i] <= right_subarray[j]:
                arr[k] = left_subarray[i]
                i += 1
            else:
                arr[k] = right_subarray[j]
                j += 1
                # Count inversions when an element from right_subarray is picked.
                inversions += (mid + 1) - (left + i)
            k += 1

        while i < len(left_subarray):
            arr[k] = left_subarray[i]
            i += 1
            k += 1

        while j < len(right_subarray):
            arr[k] = right_subarray[j]
            j += 1
            k += 1

        return inversions

    def merge_sort_and_count(self, left, right):
        arr = self.arr
        inversions = 0
        if left < right:
            mid = (left + right) // 2

            inversions += self.merge_sort_and_count(left, mid)
            inversions += self.merge_sort_and_count(mid + 1, right)
            inversions += self.merge_and_count(left, mid, right)

        return inversions

    def count_inversions(self):
        arr = self.arr
        return self.merge_sort_and_count(0, len(arr) - 1)

class IOSingleLayerDMU:
    def __init__(self,graph_matix1,graph_matix2,alpha):
        self.graph_matix1 = graph_matix1
        self.graph_matix2 = graph_matix2
        self.alpha = alpha

    def IOSLDMU(self):
        X1 = np.transpose(self.graph_matix1)
        non_zero_matrix1 = np.where(X1 == 0, np.inf, X1)
        min_non_zero_value1 = np.min(non_zero_matrix1)
        replacement_value1 = min_non_zero_value1 * self.alpha
        X1 = np.where(X1 == 0, replacement_value1, X1)
        Y1 = np.transpose(X1)
        deg1 = len(X1)
        keys1 = [str(i) for i in range(deg1)]
        values1 = [[*X1[i], *Y1[i]] for i in range(deg1)]
        data1 = pd.DataFrame(
            {key: value for key, value in zip(keys1, values1)},
            index=[str(i) for i in range(2 * deg1)]
        ).T
        X_1 = data1.iloc[:, :deg1]
        Y_1 = data1.iloc[:, deg1:]
        dea1 = DEA(DMUs_Name=data1.index, X=X_1, Y=Y_1)
        res1 = dea1.dea()['效益分析']['综合技术效益(CCR)']
        ef1 = dea1.dea()['规模报酬分析']['有效性']

        X2 = np.transpose(self.graph_matix2)
        non_zero_matrix2 = np.where(X2 == 0, np.inf, X2)
        min_non_zero_value2 = np.min(non_zero_matrix2)
        replacement_value2 = min_non_zero_value2 * self.alpha
        X2 = np.where(X2 == 0, replacement_value2, X2)
        Y2 = np.transpose(X2)
        deg2 = len(X2)
        keys2 = [str(i) for i in range(deg2)]
        values2 = [[*X2[i], *Y2[i]] for i in range(deg2)]
        data2 = pd.DataFrame(
            {key: value for key, value in zip(keys2, values2)},
            index=[str(i) for i in range(2 * deg2)]
        ).T
        X_2 = data2.iloc[:, :deg2]
        Y_2 = data2.iloc[:, deg2:]

        dea2 = DEA(DMUs_Name=data2.index, X=X_2, Y=Y_2)
        res2 = dea2.dea()['效益分析']['综合技术效益(CCR)']
        ef2 = dea2.dea()['规模报酬分析']['有效性']

        dataframeres = pd.DataFrame({'theta_before':res1,'theta_after':res2,'eff_before':ef1,'eff_after':ef2})
        sorted_df = dataframeres.sort_values(by='theta_before')
        res_df = InverseOrder(list(sorted_df['theta_after']))
        simnum = 0
        for i in range(len(sorted_df['theta_before'])):
            if sorted_df.iloc[i][0]==sorted_df.iloc[i][1] and sorted_df.iloc[i][0]==1:
                if sorted_df.iloc[i][2]!=sorted_df.iloc[i][3]:
                    simnum += 1
        print(simnum)
        return res_df.count_inversions() + simnum

class DMULayerInfluence:
    def __init__(self,graph_beforein,graph_beforeout,graph_afterin,graph_afterout,alpha):
        self.graph_beforein = graph_beforein
        self.graph_beforeout = graph_beforeout
        self.graph_afterin = graph_afterin
        self.graph_afterout = graph_afterout
        self.alpha = alpha

    def IOSLayerDMU(self):
        X_before = self.graph_beforein
        Y_before = self.graph_beforeout
        X_after = self.graph_afterin
        Y_after = self.graph_afterout

        non_zero_X_before = np.where(X_before == 0, np.inf, X_before)
        min_non_zero_valueX_before = np.min(non_zero_X_before)
        replacement_valueX_before = min_non_zero_valueX_before * self.alpha
        X1 = np.transpose(np.where(X_before == 0, replacement_valueX_before, X_before))

        non_zero_Y_before = np.where(Y_before == 0, np.inf, Y_before)
        min_non_zero_valueY_before = np.min(non_zero_Y_before)
        replacement_valueY_before = min_non_zero_valueY_before * self.alpha
        Y1 = np.where(Y_before == 0, replacement_valueY_before, Y_before)

        non_zero_X_after = np.where(X_after == 0, np.inf, X_after)
        min_non_zero_valueX_after = np.min(non_zero_X_after)
        replacement_valueX_after = min_non_zero_valueX_after * self.alpha
        X2 = np.transpose(np.where(X_after == 0, replacement_valueX_after, X_after))

        non_zero_Y_after = np.where(Y_after == 0, np.inf, Y_after)
        min_non_zero_valueY_after = np.min(non_zero_Y_after)
        replacement_valueY_after = min_non_zero_valueY_after * self.alpha
        Y2 = np.where(Y_after == 0, replacement_valueY_after, Y_after)

        print(f"X1 shape: {X1.shape}, Y1 shape: {Y1.shape}")
        print(f"X2 shape: {X2.shape}, Y2 shape: {Y2.shape}")

        deg1 = len(X1)
        degY1 = Y1.shape[1]
        keys1 = [str(i) for i in range(deg1)]
        values1 = [[*X1[i], *Y1[i]] for i in range(deg1)]
        data1 = pd.DataFrame(
            {key: value for key, value in zip(keys1, values1)},
            index=[str(i) for i in range(degY1 + deg1)]
        ).T
        # print(data1)
        X_1 = data1.iloc[:, :deg1]
        Y_1 = data1.iloc[:, deg1:]

        dea1 = DEA(DMUs_Name=data1.index, X=X_1, Y=Y_1)
        res1 = dea1.dea()['效益分析']['综合技术效益(CCR)']
        ef1 = dea1.dea()['规模报酬分析']['有效性']

        deg2 = len(X2)
        degY2 = Y2.shape[1]
        keys2 = [str(i) for i in range(deg2)]
        values2 = [[*X2[i], *Y2[i]] for i in range(deg2)]
        data2 = pd.DataFrame(
            {key: value for key, value in zip(keys2, values2)},
            index=[str(i) for i in range(degY2 + deg2)]
        ).T
        # print(data2)
        X_2 = data2.iloc[:, :deg2]
        Y_2 = data2.iloc[:, deg2:]

        dea2 = DEA(DMUs_Name=data2.index, X=X_2, Y=Y_2)
        res2 = dea2.dea()['效益分析']['综合技术效益(CCR)']
        ef2 = dea2.dea()['规模报酬分析']['有效性']

        dataframeres = pd.DataFrame({'theta_before': res1, 'theta_after': res2, 'eff_before': ef1, 'eff_after': ef2})
        sorted_df = dataframeres.sort_values(by='theta_before')
        res_df = InverseOrder(list(sorted_df['theta_after']))
        # print(dataframeres)
        simnum = 0
        for i in range(len(sorted_df['theta_before'])):
            if sorted_df.iloc[i][0] == sorted_df.iloc[i][1] and sorted_df.iloc[i][0] == 1:
                if sorted_df.iloc[i][2] != sorted_df.iloc[i][3]:
                    simnum += 1
        print(simnum)
        return res_df.count_inversions() + simnum

class IOMultipleLayerDMU:
    def __init__(self,supernetbefore,supernetafter,alpha):
        self.supernetbefore = supernetbefore
        self.supernetafter = supernetafter
        self.alpha = alpha

    def IOMLDMU(self):
        supernetbefore = self.supernetbefore
        supernetafter = self.supernetafter
        supernetdeg = len(supernetbefore)
        EIO = np.empty((supernetdeg, supernetdeg))
        for i in range(supernetdeg):
            for j in range(supernetdeg):
                if i == j:
                    EIOele = IOSingleLayerDMU(supernetbefore[i][j],supernetafter[i][j],self.alpha)
                    EIO[i][j] = EIOele.IOSLDMU()
                else:
                    EIOele = DMULayerInfluence(supernetbefore[i][i],supernetbefore[i][j],supernetafter[i][i],supernetafter[i][j],self.alpha)
                    EIO[i][j] = EIOele.IOSLayerDMU()
        EIO = np.transpose(EIO)

        return EIO

if __name__ == '__main__':
    block_matrix_seq = np.load('block_matrix_seq_new.npy',allow_pickle=True)
    ss = block_matrix_seq[0]
    print([[num.shape for num in ele] for ele in ss])
    print(np.all(ss[0][0] == 0))
    print(np.all(ss[0][1] == 0))
    print(np.all(ss[0][2] == 0))
    print(np.all(ss[0][3] == 0))
    print(np.all(ss[1][0] == 0))
    print(np.all(ss[1][1] == 0))
    print(np.all(ss[1][2] == 0))
    print(np.all(ss[1][3] == 0))
    print(np.all(ss[2][0] == 0))
    print(np.all(ss[2][1] == 0))
    print(np.all(ss[2][2] == 0))
    print(np.all(ss[2][3] == 0))
    print(np.all(ss[3][0] == 0))
    print(np.all(ss[3][1] == 0))
    print(np.all(ss[3][2] == 0))
    print(np.all(ss[3][3] == 0))
    print('=================================================')

    block_matrix_seq[0][2][3] = np.ones((19, 16)) * 0.001

    ss = block_matrix_seq[0]
    print(np.all(ss[0][0] == 0))
    print(np.all(ss[0][1] == 0))
    print(np.all(ss[0][2] == 0))
    print(np.all(ss[0][3] == 0))
    print(np.all(ss[1][0] == 0))
    print(np.all(ss[1][1] == 0))
    print(np.all(ss[1][2] == 0))
    print(np.all(ss[1][3] == 0))
    print(np.all(ss[2][0] == 0))
    print(np.all(ss[2][1] == 0))
    print(np.all(ss[2][2] == 0))
    print(np.all(ss[2][3] == 0))
    print(np.all(ss[3][0] == 0))
    print(np.all(ss[3][1] == 0))
    print(np.all(ss[3][2] == 0))
    print(np.all(ss[3][3] == 0))

     tempio_seq = []
     for i in range(1, len(block_matrix_seq)):
         print(i)
         tempio_value = IOMultipleLayerDMU(block_matrix_seq[i-1], block_matrix_seq[i], 0.000000001)
         tempio_seq.append(tempio_value.IOMLDMU())
     #
     print(tempio_seq)
    
     np.save('layerchange2.npy', tempio_seq)
