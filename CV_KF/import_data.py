import xlrd
import matplotlib.pyplot as plt

xl = xlrd.open_workbook(r'../track.xlsx')
table = xl.sheets()[3]

time = table.col_values(0)
dist_x = table.col_values(2)
dist_y = table.col_values(3)
heading_rad = table.col_values(4)
heading_degree = table.col_values(6)
life = table.col_values(5)