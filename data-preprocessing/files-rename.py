import os
 
# Function to rename multiple files
def main():
   
    folder = "C:/Users/kkotkar1/Desktop/CropAllDicoms/YoloData/GroundTruthFixedCropDicom"
    for count, filename in enumerate(os.listdir(folder)):
        count1 = str(count+1).zfill(3)
        dst = f"dicom-{count1}.dcm"
        src = f"{folder}/{filename}"  
        dst = f"{folder}/{dst}"
         
        os.rename(src, dst)
 
# Driver Code
if __name__ == '__main__':
    main()