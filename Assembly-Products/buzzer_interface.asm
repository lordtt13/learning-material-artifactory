IO0DIR EQU 0XE0028008
IO0SET EQU 0XE0028004
IO0CLR EQU 0XE002800C
IO0PIN EQU 0XE0028000
START
LDR R0,=IO0DIR
LDR R1, =0X02000000
STR R1,[R0]
LDR R2,=IO0SET
STR R1,[R2]
BL DELAY
LDR R3,=IO0CLR
STR R1,[R3]
BL DELAY
START DELAY
LDR R4,=0X000FFFf 
L SUBS R4,R4, # 0X0
BNE L
BX LR
END