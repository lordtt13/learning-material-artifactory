again:mov r1,#0dh
	  mov r0,#00h
start:mov dptr, #TABLE
loop: mov a,r0
	  movc A, @A+dptr
	  mov p1,A
	  mov A,r1
	  dec A
	  movc a,@a+dptr
	  mov p0,A
	  inc r0
	  djnz r1, loop
	  mov r1,#0dh
	  mov r0,#00h
	  sjmp loop
	  sjmp again
	  
TABLE: DB 128,192,238,255,238,192,128,64,17,0,17,64,128
  
l1:sjmp l1
end