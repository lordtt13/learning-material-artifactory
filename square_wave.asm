main: mov p0,#0ffh
acall delay
mov p0,#00h
acall delay
sjmp main
delay: mov tmod,#01h 
mov ie,#81h
mov th0,#0dch
mov tl0,#00h
setb tr0
l: jnb tf0,l
clr tf0
clr tr0
ret
end