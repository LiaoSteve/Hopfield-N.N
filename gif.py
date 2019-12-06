#------------  Use the .png of the folder to create the .gif
import os
import imageio

dirName1 = 'result/noise/'
dirName2 = 'result/unknown/'
dirName3 = 'result/both/'
Name = []
dirs = os.listdir(dirName3) #--- change dirName 1,2,3
frames = []
gif_name1 = 'noise.gif'
gif_name2 = 'unknown.gif'
gif_name3 = 'both.gif'

for file in dirs:
    frames.append(imageio.imread(dirName3 + file)) #-- change dirName 1,2,3
    imageio.mimsave(gif_name3, frames, 'GIF', duration = 1) #-- change gif_name 1,2,3
    print (file) 

