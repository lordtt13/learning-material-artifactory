ORG 0000H
LJMP 1000H
ORG 03H
MOV DPTR,#0EE00H
MOV R1,# 0EEH
MOV R0, # 00H
RETI
ORG 13H
MOV DPTR, #0DF99H
MOV R1, #0FCH
MOV R0, #066H
RETI
ORG 1000H
MOV DPTR, #0F533H
MOV R1, #0E6H
MOV R0, #0CCH
MOV IE, #85H
MAIN: MOV P0,# 0FFH
ACALL DELAY
MOV P0,#00H
ACALL DELAY 
SJMP MAIN
DELAY: MOV TMOD, #01H
MOV TH0, DPH
MOV TL0, DPL
SETB TR0
L: JNB TF0,L
CLR TF0
CLR TR0
RET 
END