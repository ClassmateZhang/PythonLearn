import random

print('猜字游戏1~20之间，请在6次内猜对。')
print('你确定是否要开始游戏？（Y/N）')
start = input()
pcNumber = random.randint(1,20)
userNumber = 0
count = 0
if start == 'Y':
    while count < 6:
        print('第'+count+1+'次竞猜')
        userNumber = input()
        if userNumber == pcNumber:
            print('猜对了！！！')
            break
        elif pcNumber>userNumber:
            print('猜小了！！！')
else:
    print('猜大了！！！')
count = count + 1
if count ==5:
    print('游戏结束！挑战失败！！')
else:
    print('很遗憾你放弃了游戏。')
