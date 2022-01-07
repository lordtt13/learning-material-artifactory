org 0h
ljmp start
org 0bh
cpl p1.0
cpl p0.0
reti

org 50h
start: setb p0.0
       clr p1.0
l: mov tmod,#02h
   mov ie,#82h
   mov th0,#0f7h
   mov tl0,#0f7h
   setb tr0
   
l1:sjmp l1
end