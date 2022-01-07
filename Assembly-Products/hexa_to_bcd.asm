mov r0,#40h
mov 40h,#0ffh
mov a,@r0
mov b,#64h
div ab
mov r1,a
mov a,b
mov b,#0ah
div ab
mov r2,a
mov r3,b
end