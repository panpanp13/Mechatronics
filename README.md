# Cross Road RECHECK
VIDEO
[![Watch the video](https://img.youtube.com/vi/nek8qwRLXHk/maxresdefault.jpg)](https://youtu.be/nek8qwRLXHk)

### [Watch this video on YouTube](https://youtu.be/nek8qwRLXHk)

-------------------------------------------------------------------------------------------------------------------------
# Background

The inspiration for our project, Crossing Road Recheck, arose when our team came across news reports about pedestrian accidents occurring at zebra crossings. Generally, zebra crossings are intended to be one of the safest options for crossing roads, second only to pedestrian overpasses. This observation prompted our team to conduct further research, through which we discovered that the Consumer Organization Council (COC) reported accident statistics from the National Statistical Office. These data revealed that between 2,500 and 2,900 pedestrians are involved in traffic accidents each year while crossing roads, with approximately one-third of these incidents occurring in Bangkok—averaging around 900 cases annually. This figure is considered significant. Consequently, our team recognized the severity of this issue and decided to make it the central focus of our project.

![image](https://github.com/user-attachments/assets/5052a073-f003-4255-a1ab-b38ae92c99ae)

-------------------------------------------------------------------------------------------------------------------------
# Related previous works

After recognizing the issue and reviewing supporting statistical data, our team proceeded to search for related prior research. We found several studies of interest, including the following:

Crossroad Accident Prevention System Using Real-Time Depth Sensing

The Use of Convex Mirrors at crosswalk to Reduce Accidents

Crossroad Accident Prevention System Using Real-Time Depth Sensing

This study employed cameras to detect the speed of vehicles approaching a zebra crossing. The system defined four specific detection zones, or "boxes," along the roadway. By analyzing the depth data of each box and tracking a vehicle's movement from Box 1 to Box 2 over time, the system could calculate the vehicle's speed. If a vehicle approached the crossing at a high speed, a visual alert system was activated. This system used four different colors—green, yellow, orange, and red—to indicate the level of danger, thereby warning pedestrians and drivers accordingly.

![image](https://github.com/user-attachments/assets/8966fe11-08b3-4d7f-adfe-07d3ead221d5)  ![image](https://github.com/user-attachments/assets/c32ca73e-46bd-48b4-bdf0-70c69398be24)


-------------------------------------------------------------------------------------------------------------------------
# Wirring Diagram

| อุปกรณ์                 | ใช้ทำงาน                                                         | ต่อเข้ากับ                                      |
|-------------------------|------------------------------------------------------------------|-------------------------------------------------|
| Laptop                  | ใช้แสดงผลการแจ้งเตือนรถในแต่ละเลนจาก Raspberry Pi             | รีโมทผ่าน **RealVNC**                          |
| RealVNC                 | แอปสำหรับรีโมต Desktop ของ Raspberry Pi 4 ไปยัง Laptop          | –                                               |
| Raspberry Pi 4 Model B  | รันระบบทั้งหมด                                                  | กล้อง Webcam และ Power Bank                    |
| กล้อง Webcam            | จับภาพรถที่วิ่งเข้ามายังทางม้าลาย                               | พอร์ต USB ของ Raspberry Pi                     |
| Power Bank              | แหล่งจ่ายไฟให้บอร์ด Raspberry Pi                               | พอร์ต USB‑C Power ของ Raspberry Pi             |


![image](https://github.com/user-attachments/assets/ad1c809b-c133-45ec-814e-08785a2f143a)

-------------------------------------------------------------------------------------------------------------------------
# List of equipment and prices
![image](https://github.com/user-attachments/assets/0bb53645-0714-4387-8e82-4bdb34c103a3)

-------------------------------------------------------------------------------------------------------------------------
# System Interface
The boundaries for road marking were determined by selecting 4 specific points to define the area

![image](https://github.com/user-attachments/assets/b843ec7f-f2bb-47fb-b5d4-d5ce580a22cc)

fill amount of road lane

![image](https://github.com/user-attachments/assets/da5c8c48-3b64-4305-a81c-311329aca4fd)   ![image](https://github.com/user-attachments/assets/9393e7c9-7fd0-4c48-8482-24956fb57072)

move the green line to fit in road lane

![image](https://github.com/user-attachments/assets/8d6aa7cd-d49f-4f12-83bd-ff4b6c481546)

screed display trafic light when it green is safe to cross a crosswalk ,but when it red mean still have car lead to crosswalk

![image](https://github.com/user-attachments/assets/7fa38237-1968-4269-9598-28e9d5c7c706) ![image](https://github.com/user-attachments/assets/eab80f6b-f6a1-4f02-afef-4de10ac18dfc)

-------------------------------------------------------------------------------------------------------------------------
# Reference
https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10392614&tag=1&tag=1




