import pandas as pd
import xlrd as xd
import os
import csv
def get_result(content):
    result = [0,0,0]
    if str(content).find('疏松')!=-1:
        result[0] = 1
    if str(content).find('裂纹')!=-1:
        result[1] = 1
    if str(content).find('夹杂')!=-1:
        result[2] = 1
    return result


#result_path = 'C:\\Users\\28929\\Desktop\\wy课题-质量统计\\F1B22-22出口管\\result\\result.xls'
result_path = r'C:\Users\28929\Desktop\data02.xls'
resultbook = xd.open_workbook(result_path)
resultsheet = resultbook.sheet_by_name('Sheet1')

i = 1
result_map = {}
while i < resultsheet.nrows :
    
    index = resultsheet.cell_value(i,0)
    content = resultsheet.cell_value(i,2)
    
    result = get_result(content)
    result_map[index] = result
    i += 1

i = 1

dir_path = 'C:\\Users\\28929\\Desktop\\xls'

files = os.listdir(dir_path)
data_csv_path = r'C:\Users\28929\Desktop\data_shell.csv'
data_file = open(data_csv_path,'w+',newline='')
writer = csv.writer(data_file)
#data1 = ['para1','para2','para3','para4','para5','para6','para7','para8',
         #'para9','para10','para11','flaw1','flaw2','flaw3']
#writer.writerow(data1)
for file in files:
    
    file_path = dir_path + "\\"+file
    workbook = xd.open_workbook(file_path)
    worksheet = workbook.sheet_by_index(0)
    nrows = worksheet.nrows
    index = worksheet.cell_value(3,1)
    if str(index) in result_map and worksheet.cell_value(29,0) == "制壳参数记录表":
        data = [worksheet.cell_value(32,4),worksheet.cell_value(32,6),worksheet.cell_value(32,8),worksheet.cell_value(32,10),
                worksheet.cell_value(33,4),worksheet.cell_value(33,6),worksheet.cell_value(33,8),worksheet.cell_value(33,10), 
                worksheet.cell_value(34,4),worksheet.cell_value(34,6),worksheet.cell_value(34,8),worksheet.cell_value(34,10),
                worksheet.cell_value(35,4),worksheet.cell_value(35,6),worksheet.cell_value(35,8),worksheet.cell_value(35,10),
                worksheet.cell_value(36,4),worksheet.cell_value(36,6),worksheet.cell_value(36,8),worksheet.cell_value(36,10),
                worksheet.cell_value(37,4),worksheet.cell_value(37,6),worksheet.cell_value(37,8),worksheet.cell_value(37,10),
                worksheet.cell_value(38,4),worksheet.cell_value(38,6),worksheet.cell_value(38,8),worksheet.cell_value(38,10),
                worksheet.cell_value(39,4),worksheet.cell_value(39,6),worksheet.cell_value(39,8),worksheet.cell_value(39,10),
                worksheet.cell_value(40,4),worksheet.cell_value(40,6),worksheet.cell_value(40,8),worksheet.cell_value(40,10),
                worksheet.cell_value(41,4),worksheet.cell_value(41,6),worksheet.cell_value(41,8),worksheet.cell_value(41,10)] + result_map[index]
        
        print(i,index,data)
        writer.writerow(data)
    i+=1
data_file.close()
