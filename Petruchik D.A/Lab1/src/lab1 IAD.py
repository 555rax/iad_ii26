
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
from zipfile import BadZipFile, ZipFile
import zlib

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

DATA_FILENAME = 'Wholesale customers data.csv'
TARGET_COLUMN = 'Region'
FEATURE_COLUMNS = [
    'Channel', 'Fresh', 'Milk', 'Grocery', 'Frozen',
    'Detergents_Paper', 'Delicassen',
]

DATA_SHA256 = 'c3d018c643565b85cee733c4a2ac76dd76e080e857cb23f0ccfcc2e15a6c17ef'
DATA_B85 = (
    'c-l>yxsD*m5k!3f{F657K5Pk+B|y**2m~=82@y+SmyY!HiWgPcv(At^(_KfNk&)Sd{^iFnUp{_&{`v9qufKn'
    'J{`&Rf+b_@GfBo&>&)>fO{?o_T|LRx2|L^0=^Oui5KED3^@#T+i|N6&||M~c;U;Os#Pd|S9_VMM1Kb5CGxs0'
    ')&J^FZNANeV{J!Q=H<XYExWTrl!y>fbXAM43;=(lC--8sj3WYk`twbFX#SX!W^{?wke=4$PSKV>$mW%MWKJ)'
    'Txdenv~}>1k?}bvG(y<o@)goo3$asayZlCiCfe_4CYbSv!@!^}8~jF@~PldaU&vH{uU9@BNHD`_op^zgp75c'
    'a7QSewr?$)%0ja0=4Q)rH^q=+xMtX9c?`$@1hfP5bo2{y>Uu<GbhBb*2?;SooVu5N&8-?RvoXm#G$ri+FH^v'
    '_T<vY#mCjzSgGjGz3AVqa8wRB?<(3^`Y5`Rp3~u<vgT9SR9d*V)}`}&Mv)3~+e^PnlFFD8@JQu8(?&;v=s}0'
    '%byU!o-X80!DHT1Fvv#aYBB@)tA39sUi|BzQRZA8KTKBFToNd!eS9ohmGns?#Ds5GXcP8C_Ub*Bok}RMW2_h'
    'vasuUSanM2pgE1y;7l;Voi#EIL3l8Wp=2M$HgF3Cynl9leLr6lcwY3N|r>U!6jYwD^=Rwa{`IAs%k?DO}Uwd'
    '<3&WY-#w&~%ehCE8Z&)leCA+87EZnlh%>8xQFCLppn{w@Fr$_3D(-6wJ3Z>7vi~=0k7DoT^N`Z)`<7B^gMc='
    'RkYO@HdXQ^rPmk|7%4utXDsc124T%Q^UABBE&(?1})1!N@{qaHkjDudL(aYx}<B}t)_+2^<o)mV+{#omZ(bX'
    'BNfYsN9GUwMHay;(sq$=oQLGE1vyB+3K!Z|t1g+x##{YBmXa}Xy}FjPBMoi+L-FB0uxF9_rv5f7{+uNl?TR('
    'UNjkdS6Dt|V6&q^Y*PDknmA_<rV@6WQSBusWd+gr%rxei&0ZdyimKx>8FNvARmSipxM#Dk3?lHQYT+Vkz+u5'
    'Y^c9vriGZbO;MhoRAfD)Lr+qOL>{gf?Y32~N|IKQHeE=^eN7O*tg*N8|dYg7PBGL+8pB_Z1`vn?yVUbs3Y)Q'
    'NnUi}WQ)Y+NBivQ>(W)h(Nvi`{hgd6(F_gJQdVu*pLdclENAGp=A|QyGJAQhDw*6SVXOp}y^i$2Ak(TEaUsa'
    '#l3e64<|Pb+>BrEdpUV($Qve2#KXBvMCY7hB8$-mt=_SmX_(V;_e{RBGc%ylO%;y*XB=Cm&>&;-Vei+Vx=T5'
    'uRJ1ylKnpzy6mxB$FFi$nW6UIN=X=u?x>s-(WRy7@TWlMc-Ez|qW_b9gfrXdBs-s6e^yM#1Q6O)Wsy;9;ihn'
    '&w>B)YLcEjg8aAV_A&b}XdrCMnVKXTk$dOYVMOQ3}3_8JB1i3-@ov1?&d7qO0q+tTC&g29^qLV1gDv^Tc<RH'
    '>}Cbe}gF8#lrBJWeKtrFHstgW*vQs^in-5|I6PABrrr5H#uw~3W7hpv>Wt*hiO<0XlgZbXqBgKtWc_ED0S8N'
    'HEQh^J|?;B~fLH(8N9A(yl5>lexpC}7{{A*<k&x`KAmxh7(h``V-8z~p1uUF%D}kN6cGu&9W1R(zB4GHF>cu'
    'gUVtJ*ZHG;^|PVM$!Xm$|Jk%Pk|<sEbEq6=_qnEXFqw#TF$`diB&u1S}amBo2;Bj^I{{3hjUggaxz)7Icw9='
    'BOn*djR>m1Eg^TZk`9ggwRu(Cj2^PASr6)r3j+Y!IIW%=5hXeiNO@6b%UCW6wq3JcSvnq7Cs5EupyZg1B4%W'
    'G9Tl&`T-Y=5A7QPVxFyJnrwltqS0EzptR0BQ*<Lr<AQ{0U1!LzRX#qG8KxBgL(1`LDKCuak8DWLg^lBxhNJ#'
    'x5U}Q^A``(k_QOj1GNC>)C_F)1uDdDrf1gI!h7KW(+n0us!MAF!W9LC8M{J5Hal-f#&Vu6<X0FY<}J*+JV&J'
    'k#{;LZeK%F`A|PS4>YSaNz{0SH7`Q!ZQpK7v{J)+&Igz>1}8Wsg~#{R)h>7m`V`xSeVwcYt;$&n#9Q!Z`R4a'
    'j5nHSIRa7WNo1zTS5IXFnQC|PR(wNrAW%Hk^dT4G=O3T<P$;^b=Qff;OKEpL}3!gr&uVMfOL$@<C_r$HXy%D'
    '882op?N>{B?SzZfSO|oSGbIc`k9KQ9;ba8P?gJQK3JxZ7Mk!h$=~yLP3>ZL0FY-ibTcX4~!Vu~Ssgf7)Sup^'
    'emV6D(#-Yn#JPNuG{~OVp>?ipupUJ=hmblvuEW^lpfnIcCSqKTJbIUj1$-kz9vQirN*Xe}q^h^pPbv>YT2u{'
    '#7CBKd!?HnsgG)0vn2c`lcLKZ5y)(H(teSuBVvhdnt0cT+eE_*<BD76_A5)!2HPMRW2ZO}nH(wzhoIw6y64o'
    '3E@FUD;a5cVy@bgmOL0gkMwR@U6pQI6FO3_&1@@HWy|2h6c3K%{73Fc)*=sNtWDiwqE%(u0;6lVGbj5vF5zC'
    'fDU5Q5S=KHW5^hU}Vp5!J}BCzb$PcQ6lIZu}I#aGtP;ucw~WraQ2yA$u-Qa06C^ETrtK@<diNVf&--}lW<x6'
    'FnZ&FCCW}6gH*K9C<uLKmql3K`@qB)fhLvLJHa+%&)4b%X8=6fL~zE5o(HybX+*399$6HC2E~MW)eXwIS8l4'
    'Wse%mrc%9Ts;!&A@kD*#PowA`Cik;+X&=NHUb)rnYxgG^z=_-S&B-SW~#^3@hZF3x=c!WgVG&u_(Z&-;;GTL'
    'ac-E%6~bYBsOjmZOYd$OlQpc{pl4M>~$CWmEr-8odZNd$S32yj{DKTyC%!;vVcnT0M$2}4Rnys8qhi550O9$'
    'S{ysp5IGge8JG$Rq=eCFG|fi2qq(V=6N?*kqwWg7%=q(J6D3Qj%0(lv)jpqUeR<WAmMyl~IN4xl2lb@mlU3c'
    'XM*(RHeEYB1mY5tF0-Ep@yJdks7CT*MNdI0;-rm-HQ_3l@_YeF>48J4|Ins%_L86u2vdItVg00l2m<(^$Un8'
    'T1j6<*kSt4t11!DAcyU&z!vjLK#H5NA{1WDMyhj>FGB<9S}QDy6!27;Fk8v7J<d?^cE?$jH>@t9f(HAQWYzh'
    'p`3S@?Ary}fmK>;rte2hk03?fYv?7Vunie^`8mQvKI5<pLiNc})o)o1PYMdsQh(Z#pLP!ekgCMohl3>2=W>P'
    'BLKP!__!Au7rV8r4$5Dn$!uLOl%J~_u5egX|58_Mn|&?pv$?!R3BJBo!xjRNPjOneP=i-2*+7k*(5LN$wWg('
    '!mmny+iGBwV0Ocns$>SujXgu6B>n2AAnL(^$sE)jnFn?pT;KG=(D{na?MN<s=&`Y-qSTo09r+zfN1%0iI}g;'
    'R|3PvVblh1(9+umN%S$8Y0F0rHP<Z%)*RLo})D0*mfePja8!J(ZQVNR5$1sighdLAT>&+JTVcZOeKa{K?D{T'
    ')vqXVPmm~Dl3^V?c{!hs5x6L<-~feUs`_%$f&q}rb5(o9Y3dKjb0RiCWH16<WvSv}q9`&w1bl|-EuR?|0#@m'
    '$TgHlo4f|H$B!JJ0NE){o`0mCcsCnd-MB@k-WgiBX05r_@ATXh?9k6K`Om`DmLJzWdh1+msRkF=hL~#(PW^z'
    'BNS1K(bYRsaU<ban=-vg`S><Y!BTj;Mgaa64+6)x0anq7#27hqVhR=HD<AMI9~8$hrLG|Q18ATmA&-P)D>U?'
    'vF*_Ml)}2~u-4=Xg+~95EX%@4%Iv3TEuS&M78NU?!o|?@%<7a!jq}Jjt1Gl7T8RER)WysJ^`!9!A89>a)Vc3'
    'S}+elWEuqF##F!rsAq(Fo=A53|@q)kz_D;II|EcT;w){U)hjqHg776?;%$lZ26=KvX;6*ZaPm*b#gXIvgvfm'
    'ga9^3{cXxl+eK>EHf9PZpc~f&TwHj<Tcb5spkV4nX2gXf$^*A$pHLKvv9i$y?-kYwQd>w(#<34qRKd={alm~'
    '{(7G&(`f8#eSZozb_Yt3lc!Ri+Q%7h>ChjRUOPm;-jSw9cCxe&f6ZV2k#f}%?SE|v<FL4bo+aQ8vTbQcWMl('
    '?sKx;L#8}haXhNGLj3n)2F+9tD)lHU?iHek%{4SuQA!xeP%(O3wECL6O|WH`uTFaQ?XK7oRxL_-I{QN`@^q>'
    '^H@YHVOeDW8CN)<KS>GU@`tpOU~NMLpLQ@*yJ&sk=~4b?`@`2$^S>njFsxArj7B)-A8w+s={CGY!&PIgNgTl'
    '&;IrMuF&Y_{nD{rGD3JP-HR-r3i>tIlqq9+j#~u5tt{0s`xp~5<EFB11Af3mg1@QVD32DI;Xf~IfBy0`{1xm'
    'Nj?u+qwxV6UG&ow$)>q+n2;+_6R|K2s@OKwKSj9vE%*HlQmPXMPcQ<TBN1am=_;i|KuAo7s-^`p%#2B_GenN'
    '5qEZ{6+gOI;&3%D;7b%0%3T9qVQLe3X(JleH$^`Ctrjx8qEtw(%%9Hx%3`07}R~j88)C?<HO<T(-?zJgJMkh'
    '_mGui|1<_5;xp*d`rf`CX1idTLE<P>*oR!FZJ<O9i6_qjVr1)3nGTio1*STnUmVdr<8N_uD!@|2|B4_75)Xw'
    '<AyT{(+1a?{DOFcUJ+huv!ENUJNo?R^5NF%&)4Q!YqD@SD^0fy-dvUxwzY@jz;bf&y@z7cXR=A+X85_9PQN^'
    'kSeL^QB<Wi{vw)E}PDwR7B07f8P@pQgSUW{<_s8*UnHwcY_2;-yzvIrCKHx6}N!<Vc7DakBS*>7_d^@<iUGl'
    '7q~qFB~&!_-pmN4`~HFl)iJpnEkV*HcD7ggbSDn69N#2S5IZhaa=nJ^LKR#Jt7cBaWWGS1+&i3pW0=#u09g)'
    '<6)t}{+<aP`8%U0)k?k7@CHxK0B&cmrNr^E`JR*!5*F3A@7AB!x)&u)@kDCH_irMq&Acoiva~^^!=X4M}LdN'
    'pMbsfao6uzQqbXWzGPE>2-DnbSVF#yvDj_$=J5|l*fsvHWH(rg~%YK0K%*RC8?aS=04D3}oeQGRM~+#v2@R`'
    'T<PJr<f*Qt%t?3%yoq&`$h-i~=uX+xL+HPdlOYVYY~ubo4Uy^`7Ys`5;^(ue&hhs6~6D%ymzzjMY!g>?r~Nx'
    'Ge8IO6^YWgZza-)8_ro1tc!wIK4%vr0y(AzxF!0p+KE30jEjvWK>cjCJE${5j+fmudDFH9#5?27Q-D=w?}L3'
    'rl=sj7R(>5tt$^ISGwZp5jTWeGHTwTyxl1R9W%$qw&Cc8(eqs-pW;?wisquSLnG^EpaYa}9C%1+NFz7nbE?n'
    'GIl~TJeh4TU+0r+m*Q&i{xGZ*|+SENqcdbb1o7VNU5U56@TaP8sS&*F}8HOw({BgZ@@^>|{UjPBZcGpC9#tW'
    '=qq7!%!_q{wMVX7~;rm2@m-hf4t)f*L{5-5R4fuMIPZSelRXvt^LK>>%E70DBaBgf?nC=`0!CMBA{hD!FGc4'
    'B^|*&%j|;f%sA29QV#%T4SW$(YDz&}1^WF#Xa@wY=d&b;LF!#xH69GPvSq!}!D%I{{lyY1T}i!V?kV!>yF9F'
    'is1G2nX-r(k8Wt5+;)?2kKx<z^P*Mc#mO)&fCwRjL0T&`gJJEq6Nb~9M>h!eMpfP+`~uGAI#1Aa|axrD_WvR'
    'I1@_qQVgPH>gHOGp3N|6nGZux2}jJDJNl{gp>$^|LPEAHIMegd{@97W7m8j6;YRpGf?$d8DLfK3xlxKXI9Bk'
    '$i>)Uu&_NYx(Br0^Gy$RsUeTjH!fgyY$rU`)0~pBZv&#k1Na;L`K$EvMT@u$(eqzjMdnAM~>ej4db4$@(>>i'
    '659_z^NNhQZhfJF<L?GFCU&gzNjhE3<qFzl0$1%A8cPw8-h(15{P9sbeHQ^$k>Cu)RSUgp1cqeJ&H7exD@wN'
    '9_<ZOM@u`;i&!7!@FSas>^_qz51YzCS}cQtd`74bWb)89N%BRA>f!FSiZRR5^tL8d*I}&sU7~7Ugc?FNeo{v'
    'Z*PobZBMx8TXYvwO}rvNe$sly7yMCCpP94_5erY<B&qx^#O~hAOv>fIRtyi?*bK2P66SD4^5IvXrz^UI_8-K'
    'w$2iZe%2jQLxc@9IkS1R^TeR+R1k$>888scS%G@f0Y3Dx8l4nScRJj<5#c`XXIIDr95CF{C%z4Xuhh>Y-E@;'
    '752U0MzwSy#&!>A?&fHFimfH%sTnBq5SKdX+!P-Wi_+8r`eenQ5oCHVbjLjsB6TW+;Enl^mT9=!-w;}<vq+@'
    'FiX*9QYm#07A<IL9!V}nKyX~3aL6NQ*4bVp)?K<?nzex4ZH%`3w&8BgQpVBgd6Eho9BVb0Oz5yZdJ!#K4SOU'
    '!h78s&^RG7tb1+PAT33)+CaQjPeQAs<1^#QD9iTjxZ0rhTZHh(PlM+;UdgAa)wQ-lv&OUmlsdw@sEL=w#C6$'
    '#kR^NzcroCEwikCHE<Rkcw$ZP`uWDI#;?+cJ?yZ1NM>c%t9WNWYU6hLI#Z7a)u)%qK>dbJpkdxG(qj|dFs#8'
    'P24NhQ#m9gprZm`4Q&~=Q^@F{J$Rnsp%gMbJ)hYFqoW(tVy|<h#8hM{1_7b}&QPS!QQ-rfW}2CWxJ$(qb~Ak'
    'gW`VXF?wH&)isv&uzfhPvEY(d(!;GsKS`rTT@C}vTvjC4r7${Xn+TorAX;09RB2L$y9!OjWZr&B*@PMV98O('
    'JzAx*hi;M(5h+diqZzwaN)o@fVS(apRN`0EY}f*~j5_RKoRZw%(cKuA;W!J>IMu1rMR<4m{Sv*;N*HPiyknw'
    '((A6hpbQrMgEaiGhx2#}W&&$?Ec!_{Bv4e7~FiChSdIP=l=Nf2087d3N)d(^u97p_W`<B|1-bVlsr)fb5_nH'
    'm^x<d_FUgP#DKDU)T`#xDQmrqM6Dm?)dp!jKhr8sYuqDlG$#f&?neI?Rq|x0SnVsWHFW6*{zP?e(C<sgQ0){'
    '!$ynWM=GRO(h2b@u<L0z_WemDkxSP_ImO*Tr3j;W3YwSbv>2!P1$PWQweX3TvL`IyT^7N}l-hDW3Z{HAfwx^'
    'S!#fpEX&4~n+~XPQ+L_lGLN*b@oFT$Xk(z5eGz_FJK3QF0!m7RFgs`Ae6T2kX+2VOOKrtZUF4Mg#PO-Ox8w^'
    'f`>3gTUaYYX$mSQZ~jCx4?`R!L?pBWqO9zZ6EL)=AcnMr8`%g~Wy$Qllj!7$6;E0u{$>gLLRcxI2{F26$os2'
    '!kCP7B)t3X-|F__9LQq2Tu^UTi=fBP8M$4`@xwLZinE#(>0&u#ll;9CzsuG*feaVtX#pQ@m8T$qgY7<e$TtQ'
    '(^q8FsWWJa#HAvliSqUMha@svfAGb=6dXMobJ4*N6e(nT;7H+(}vX3V3KWDq{-Ia(O`vcyrz4WF`br*DPcE}'
    'K&XQ>uy7>9%|;WY<~YLk_Xx$Ic6e+Gu@YCq{OM#+PjqKw1}8hgB;ceIu)jEPb#JcEUG(rlnK0bsJhcLBYjca'
    '^D#@i^C0R62wt{`ShrnYFf~j&852!GB5$JmSoxqNYh!UYL&#f^uc#1oEXi;n^ExJ!Iu{Y2RWEzA=y?G7%=Gl'
    '@y&UCH82afpS-oDytVpHC@j(}%-W{M$QIy;^SHV4s%XMGtVYSA%PX4>u*5pSAdrsq4U2BIZHBhx;&PYv)4XW'
    'y8QPR*_E7M%bHn`~wz&^fo<XRKtLFBpcAL;CE;raNanJk461g`o>V<qQ`od&sz&zX&kT?4Y#Y(RJ5*jMXBMh'
    'W#V+`VQhG^DB8B@R(ezcVDRI@=n_DyIl-l>6R`}iumQ|nz1ltVKM5W`^Iqejt)8BSgfPVd=iLT)v%5CB!<+>'
    'jjZ4)s5;-GZC6haRii>c7<7EWUZaEVemV^>5(hc#SO&A_;fSO`h+SByhhE%vWB(q~T{?G@P=(oI+PgjcxH7>'
    '%%-$@gP_Q$neutxkw}~RyhNr!@E%fp9_nCupJJB7C;pQ+T5{?%0&2FB00!USk$AoS#Uf%zXa&@<Hj?KCsM=K'
    '3ec;^v4b5mg>DWkUghHq1F!;eEmZ?EHUxDT72?}+;2xj6E*$xHJ?=?!SppwrB#(Q7RRGj@g$b$1h!c<Ik;Ot'
    'w5V*bzDKfmMmFQgXK7D;R35cSMY#UW)xnb!Sp)yBpr4)L$Ot@o+TO;yR0r^5P0UgXw(k(wQOMr)xrwFh;9pb'
    '(dy=f|p%^M(H%07t<4VRgaMk#(VSd0Nfb`y+n>%Y|zxhLxa`$^t2Qs4G)t~Oo^LM0>_%=6pAlzo*9fOF^aR?'
    '+M;msgi@q|={YZ#3w&Ba0R%e}6be)$9WZYO*rl1`in71oav4r|Rt>Vt1Apuj1~{?oflycPhTarNM3yxpxcTg'
    '9pF#j84p=Z{2>Tss@E04+m0;?pRIcX(opGf~G8c}w+)Fu=ktTd2Z!drK<JuVW@|8M_SP$Z<0RubJcvv7Q?{J'
    '^=@PfXARcdgA+jL=a_2F;n)|;LYCLaXAUX%lon%fQQ{kjU52F0ibo~{S6PcHEy)a{0caZm9;Im}G<{O~(A(E'
    'Qa4<j6LM<lN$6vv=|fn6#@)SkJvNUaz$1A;ouM37AFto77F<RVsycy7cwM!}E}6jj_}M`sF*B8k4+h_=7`!g'
    ';e#l2Vqi>dbF|qaTYV9e3V7;ZC>=cutytucWpcT^@UL$?B6>?TOa~XB5t069{WV+cOoPE13mYMnuXG^Hy2m?'
    '{UQLRk$&bjz3mAc=3&?S<&-%rK4H;OT*G73hhh%)kQLUcKbi4w3?G_@Z0`u|869f=V8PA6YAT_*B{ko#0{5L'
    '`u%a`w)qFA~_YEk->?IZf?MxzhKxgED{{!eq34H'
)


def read_data(path=None):
    if path is not None:
        path = Path(path)
    if path is None:
        raw = zlib.decompress(base64.b85decode(DATA_B85))
        if hashlib.sha256(raw).hexdigest() != DATA_SHA256:
            raise ValueError('Нарушена целостность встроенных данных.')
    elif path.suffix.lower() == '.zip':
        with ZipFile(path) as archive:
            names = [n for n in archive.namelist()
                     if Path(n).name == DATA_FILENAME]
            if len(names) != 1:
                raise ValueError(f'В архиве нужен один {DATA_FILENAME}.')
            raw = archive.read(names[0])
    else:
        raw = path.read_bytes()
    frame = pd.read_csv(io.BytesIO(raw), sep=',', header=0, encoding='utf-8-sig',
                        na_values=['?', 'NA', 'NaN', 'nan'],
                        on_bad_lines='error')
    frame.columns = frame.columns.str.strip()
    expected_columns = [TARGET_COLUMN, *FEATURE_COLUMNS]
    if (len(frame) < 3 or frame.shape[1] != len(expected_columns)
            or set(frame.columns) != set(expected_columns)):
        raise ValueError('Ожидаются >=3 строк и колонки: '
                         + ', '.join(expected_columns) + '.')
    frame = frame.apply(pd.to_numeric, errors='raise')
    labels = frame[TARGET_COLUMN].to_numpy(dtype=float)
    if not np.isfinite(labels).all() or not np.isin(labels, [1, 2, 3]).all():
        raise ValueError('Region должен содержать 1, 2 или 3 без пропусков.')
    x = frame.drop(columns=[TARGET_COLUMN])[FEATURE_COLUMNS].to_numpy(dtype=float)
    if np.isinf(x).any():
        raise ValueError('Бесконечные значения признаков недопустимы.')
    missing = np.isnan(x)
    if np.any(missing.all(axis=0)):
        raise ValueError('Нельзя заполнить полностью пустой признак.')
    medians = np.nanmedian(x, axis=0)
    x = np.where(missing, medians, x)
    return x, labels.astype(int), {
        'dataset': 'Wholesale Customers',
        'target_column': TARGET_COLUMN,
        'feature_names': FEATURE_COLUMNS.copy(),
        'sha256': hashlib.sha256(raw).hexdigest(),
        'missing_by_feature': missing.sum(axis=0).tolist(),
        'rows': len(x), 'features': x.shape[1],
        'class_counts': {str(int(c)): int(np.sum(labels == c))
                         for c in np.unique(labels)},
    }


def preprocess(x, scale='standard'):
    if scale not in ('standard', 'center'):
        raise ValueError('Шкала должна быть standard или center.')
    mean = x.mean(axis=0)
    std = x.std(axis=0, ddof=1)
    if np.any(std == 0):
        raise ValueError('PCA этой работы требует непостоянные признаки.')
    divisor = std if scale == 'standard' else np.ones(x.shape[1])
    return (x - mean) / divisor, mean, divisor


def manual_pca(z):
    covariance = z.T @ z / (len(z) - 1)
    values, vectors = np.linalg.eig(covariance)
    values = np.real_if_close(values, tol=1000)
    vectors = np.real_if_close(vectors, tol=1000)
    if np.iscomplexobj(values) or np.iscomplexobj(vectors):
        raise ArithmeticError('Значимая мнимая часть при разложении ковариации.')
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    if values.min() < -1e-10:
        raise ArithmeticError('Ковариация имеет отрицательное собственное значение.')
    values = np.maximum(values, 0.)
    if not np.allclose(vectors.T @ vectors, np.eye(z.shape[1]), atol=1e-9):
        raise ArithmeticError('Базис eig не ортонормирован.')
    return values, vectors, covariance


def canonical_signs(components, scores):
    rows = np.argmax(np.abs(components), axis=0)
    signs = np.sign(components[rows, np.arange(components.shape[1])])
    return components * signs, scores * signs


def analyze(z):
    values, vectors, covariance = manual_pca(z)
    total = float(values.sum())
    projections, report = {}, {}
    for k in (2, 3):
        basis, scores = canonical_signs(vectors[:, :k], z @ vectors[:, :k])
        reconstructed = scores @ basis.T
        estimator = PCA(n_components=k, svd_solver='full', whiten=False)
        library_scores = estimator.fit_transform(z)
        library_basis, library_scores = canonical_signs(
            estimator.components_.T, library_scores)
        library_reconstructed = library_scores @ library_basis.T + estimator.mean_
        discarded = float(values[k:].sum())
        sse = float(np.sum((z - reconstructed) ** 2))
        library_total = float(np.var(z, axis=0, ddof=1).sum())
        report[str(k)] = {
            'retained_percent': float(100 * values[:k].sum() / total),
            'lost_percent': 100 * discarded / total,
            'discarded_variance': discarded,
            'reconstruction_sse': sse,
            'reconstruction_mse': sse / z.size,
            'theoretical_sse': (len(z) - 1) * discarded,
            'sklearn_retained_percent': float(
                100 * estimator.explained_variance_ratio_.sum()),
            'sklearn_lost_percent': float(100 * (
                1 - estimator.explained_variance_.sum() / library_total)),
            'sklearn_sse': float(np.sum((z - library_reconstructed) ** 2)),
            'eigenvalues_max_error': float(np.max(np.abs(
                values[:k] - estimator.explained_variance_))),
            'projector_max_error': float(np.max(np.abs(
                basis @ basis.T - library_basis @ library_basis.T))),
            'scores_max_error': float(np.max(np.abs(scores - library_scores))),
            'basis': basis.tolist(),
        }
        projections[k] = (scores, library_scores)
    return {'eigenvalues': values.tolist(),
            'ratios_percent': (100 * values / total).tolist(),
            'covariance': covariance.tolist(), 'dimensions': report}, projections


def visualize(results, projections, labels, out=None, show=True):
    import matplotlib
    if not show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11})
    styles = [(1, '#2264aa', 'o'), (2, '#dc751a', '^'), (3, '#39854a', 's')]
    for k in (2, 3):
        fig = plt.figure(figsize=(11, 4.7), layout='constrained')
        data_pair = projections[k]
        mins = np.vstack(data_pair).min(axis=0)
        maxs = np.vstack(data_pair).max(axis=0)
        span = np.maximum(maxs - mins, 1e-9)
        for panel, (name, scores) in enumerate(zip(
                ('NumPy eig', 'sklearn PCA'), data_pair), 1):
            ax = fig.add_subplot(1, 2, panel, projection='3d' if k == 3 else None)
            for cls, color, marker in styles:
                points = scores[labels == cls]
                coords = [points[:, j] for j in range(k)]
                ax.scatter(*coords, color=color, marker=marker, s=23,
                           alpha=.8, label=f'Region = {cls}')
            ax.set(xlabel='PC1', ylabel='PC2', title=name,
                   xlim=(mins[0]-.06*span[0], maxs[0]+.06*span[0]),
                   ylim=(mins[1]-.06*span[1], maxs[1]+.06*span[1]))
            if k == 3:
                ax.set_zlabel('PC3', labelpad=3)
                ax.set_zlim(mins[2]-.06*span[2], maxs[2]+.06*span[2])
                ax.view_init(elev=22, azim=40)
                ax.set_box_aspect((1, 1, .85), zoom=.82)
                ax.tick_params(labelsize=9, pad=1)
            else:
                ax.grid(alpha=.2)
            ax.legend(fontsize=9, loc='upper left' if k == 3 else 'upper right')
        keep = results['dimensions'][str(k)]['retained_percent']
        fig.suptitle(f'· Wholesale Customers · PCA {k}D\n'
                     f'Сохранено {keep:.2f}% дисперсии; классы — Region')
        if out:
            fig.savefig(out / f'pca_{k}d.png', dpi=200, bbox_inches='tight')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    positions = np.arange(1, len(results['eigenvalues']) + 1)
    axes[0].bar(positions, results['eigenvalues'], color='#2264aa')
    axes[0].set(xlabel='Номер компоненты', ylabel='Собственное значение',
                title='Спектр ковариационной матрицы', xticks=positions)
    cumulative = np.cumsum(results['ratios_percent'])
    axes[1].plot(positions, cumulative, 'o-', label='Сохранённая дисперсия')
    axes[1].axhline(95, color='#dc751a', ls='--', label='Уровень 95%')
    axes[1].set(xlabel='Число компонент', ylabel='Доля дисперсии, %',
                title='Накопленная объяснённая дисперсия', xticks=positions,
                ylim=(0, 105))
    axes[1].grid(alpha=.2); axes[1].legend(fontsize=9)
    if out:
        fig.savefig(out / 'spectrum.png', dpi=200, bbox_inches='tight')
    if show:
        plt.show()
    plt.close('all')


def self_test():
    import tempfile
    x, labels, info = read_data()
    assert x.shape == (440, 7)
    assert info['feature_names'] == FEATURE_COLUMNS
    assert TARGET_COLUMN not in info['feature_names']
    assert info['class_counts'] == {'1': 77, '2': 47, '3': 316}
    assert sum(info['missing_by_feature']) == 0
    for scale in ('standard', 'center'):
        z, _, _ = preprocess(x, scale)
        report, _ = analyze(z)
        tolerance = 1e-12 * max(1., np.abs(z).max())
        assert np.allclose(z.mean(axis=0), 0, atol=tolerance)
        if scale == 'standard':
            assert np.allclose(z.std(axis=0, ddof=1), 1)
            assert np.isclose(sum(report['eigenvalues']), x.shape[1])
        values, basis, cov = manual_pca(z)
        residual = np.linalg.norm(cov @ basis - basis * values)
        assert residual <= 1e-12 * max(1., np.linalg.norm(cov))
        assert np.allclose(z @ basis @ basis.T, z)
        for r in report['dimensions'].values():
            assert r['projector_max_error'] < 1e-9
            assert r['scores_max_error'] < 1e-9 * max(1., np.abs(z).max())
            assert r['eigenvalues_max_error'] < 1e-9 * max(1., values[0])
            assert np.isclose(r['reconstruction_sse'], r['theoretical_sse'])
            assert np.isclose(r['reconstruction_sse'], r['sklearn_sse'])
    raw = zlib.decompress(base64.b85decode(DATA_B85))
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / DATA_FILENAME
        path.write_bytes(raw)
        assert np.array_equal(read_data(path)[0], x)
        archive = Path(tmp) / 'wholesale+customers.zip'
        with ZipFile(archive, 'w') as f:
            f.writestr(DATA_FILENAME, raw)
        assert np.array_equal(read_data(archive)[0], x)
        frame = pd.read_csv(io.BytesIO(raw))
        frame[TARGET_COLUMN] = frame[TARGET_COLUMN] % 3 + 1
        frame.to_csv(path, index=False)
        same_x, changed_labels, _ = read_data(path)
        assert np.array_equal(same_x, x)
        assert np.array_equal(changed_labels, labels % 3 + 1)
        frame = frame.loc[:, frame.columns[::-1]]
        frame.to_csv(path, index=False)
        assert np.array_equal(read_data(path)[0], x)
        frame['Fresh'] = frame['Fresh'].astype(float)
        frame.loc[0, 'Fresh'] = np.nan
        frame.to_csv(path, index=False)
        imputed, _, meta = read_data(path)
        fresh_index = FEATURE_COLUMNS.index('Fresh')
        assert meta['missing_by_feature'][fresh_index] == 1
        assert imputed[0, fresh_index] == np.median(x[1:, fresh_index])
    print('Все проверки пройдены: данные, PCA, потери, реконструкция, пропуски.')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                         formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', type=Path,
                        help='Путь к ZIP или CSV; по умолчанию данные встроены.')
    parser.add_argument('--out', type=Path,
                        help='Папка для графиков PNG и results.json.')
    parser.add_argument('--scale', choices=['standard', 'center'], default='standard',
                        help='standard — стандартизация; center — центрирование.')
    parser.add_argument('--no-show', action='store_true', help='Не открывать графики.')
    parser.add_argument('--self-test', action='store_true', help='Выполнить проверки.')
    args = parser.parse_args()
    if args.self_test:
        self_test(); return
    try:
        x, labels, info = read_data(args.data)
        z, mean, divisor = preprocess(x, args.scale)
        results, projections = analyze(z)
    except (ValueError, OSError, ArithmeticError, BadZipFile) as exc:
        parser.error(str(exc))
    print('Лабораторная работа 1 — вариант 10: Wholesale Customers')
    print(f'Объектов: {len(x)}; признаков: {x.shape[1]}; '
          f'классы Region: {info["class_counts"]}')
    print('Признаки PCA: ' + ', '.join(info['feature_names']))
    print(f'Заполнено пропусков: {sum(info["missing_by_feature"])}; шкала: {args.scale}')
    print('\nКомпонента  Собств. значение   Доля, %   Накопленная доля, %')
    cumulative = 0.
    for i, (value, ratio) in enumerate(zip(
            results['eigenvalues'], results['ratios_percent']), 1):
        cumulative += ratio
        print(f'PC{i:<8} {value:15.9f} {ratio:10.6f} {cumulative:20.6f}')
    for k, r in results['dimensions'].items():
        print(f'\nk={k}: сохранено {r["retained_percent"]:.6f}%; '
              f'потери {r["lost_percent"]:.6f}%; SSE={r["reconstruction_sse"]:.6f}')
        print(f'sklearn: сохранено {r["sklearn_retained_percent"]:.6f}%; '
              f'потери {r["sklearn_lost_percent"]:.6f}%')
        print(f'Расхождение проекторов: {r["projector_max_error"]:.3e}; '
              f'координат после выбора знаков: {r["scores_max_error"]:.3e}')
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        import sklearn
        results.update(data=info, scale=args.scale, mean=mean.tolist(),
                       divisor=divisor.tolist(),
                       versions={'numpy': np.__version__, 'pandas': pd.__version__,
                                 'scikit-learn': sklearn.__version__})
        (args.out / 'results.json').write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    visualize(results, projections, labels, args.out, not args.no_show)
    if args.out:
        print(f'Графики и JSON сохранены: {args.out.resolve()}')


if __name__ == '__main__':
    main()
