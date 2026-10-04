import {AbsoluteFill, interpolate, Sequence, spring, useCurrentFrame, useVideoConfig} from 'remotion';
import {Grain, NumberTag, Orbit, Progress, RevealText, TimelineLine} from './components';
import {colors, font} from './theme';

const scene: React.CSSProperties = {background: colors.paper, color: colors.ink, fontFamily: font, padding: '70px', overflow: 'hidden'};

const Intro: React.FC = () => {
  const f = useCurrentFrame();
  const scale = interpolate(f, [0, 100], [0.75, 1], {extrapolateRight: 'clamp'});
  return <AbsoluteFill style={scene}><Grain/>
    <div style={{display: 'flex', justifyContent: 'space-between'}}><NumberTag>AN ORIGIN STORY</NumberTag><b style={{fontSize: 22}}>HUMAN × IDEAS</b></div>
    <RevealText delay={6} style={{position: 'absolute', left: 75, top: 260, zIndex: 2}}>
      <div style={{fontSize: 164, fontWeight: 900, letterSpacing: -10, lineHeight: .8}}>THE HUMAN</div>
      <div style={{fontSize: 164, fontWeight: 900, letterSpacing: -10, color: colors.coral}}>SPARK.</div>
    </RevealText>
    <div style={{position: 'absolute', right: 240, top: 330, transform: `scale(${scale})`}}><Orbit radius={200}/><Orbit radius={125} speed={-1.4} color={colors.mint}/><div style={{position:'absolute', left:175, top:175, width:50, height:50, background:colors.ink, borderRadius:'50%'}}/></div>
    <RevealText delay={42} style={{position:'absolute', bottom:120, right:75, width:510, fontSize:30, lineHeight:1.35}}>A 3.3-million-year relay race—<br/><b>from stone to silicon.</b></RevealText>
  </AbsoluteFill>;
};

const Stone: React.FC = () => {
  const f=useCurrentFrame(); const {fps}=useVideoConfig(); const hit=spring({frame:f-35,fps,config:{damping:7,stiffness:180}});
  return <AbsoluteFill style={scene}><Grain/><NumberTag>01 / TOOLS</NumberTag>
    <RevealText><div style={{fontSize:96,fontWeight:900,letterSpacing:-6,marginTop:70}}>WE SHAPED A STONE.</div><div style={{fontSize:96,fontWeight:900,letterSpacing:-6,color:colors.coral}}>IT RESHAPED US.</div></RevealText>
    <div style={{position:'absolute',left:180,top:430,width:520,height:360}}>
      <svg viewBox="0 0 520 360"><path d="M65 300 L150 40 335 70 465 240 315 325Z" fill={colors.ink}/><path d="M150 40L205 170 65 300M205 170L335 70M205 170L315 325M205 170L465 240" stroke={colors.paper} strokeWidth="4" opacity=".35"/></svg>
      <div style={{position:'absolute',right:-70,top:120,width:170,height:170,border:`5px solid ${colors.gold}`,borderRadius:'50%',transform:`scale(${hit})`,opacity:hit}}/>
    </div>
    <RevealText delay={30} style={{position:'absolute',right:120,top:500,width:680}}><div style={{fontSize:72,fontWeight:900}}>3.3M YEARS AGO</div><p style={{fontSize:28,lineHeight:1.45}}>The earliest known stone tools predate our own species. Innovation began not as a flash of genius, but as <b>knowledge passed hand to hand.</b></p></RevealText><Progress index={1}/>
  </AbsoluteFill>;
};

const Farm: React.FC = () => {
  const f=useCurrentFrame();
  return <AbsoluteFill style={{...scene,background:colors.gold}}><Grain/><NumberTag>02 / AGRICULTURE</NumberTag>
    <RevealText style={{marginTop:75}}><div style={{fontSize:118,fontWeight:900,letterSpacing:-7,lineHeight:.9}}>WE PLANTED<br/>THE FUTURE.</div></RevealText>
    <RevealText delay={25} style={{width:600,fontSize:30,lineHeight:1.5,marginTop:45}}>Around 10,000 BCE, independent communities began domesticating plants and animals. Surplus food made room for <b>cities, trades, records—and inequality.</b></RevealText>
    <svg style={{position:'absolute',right:80,bottom:100,width:800,height:760}} viewBox="0 0 800 760">
      {[0,1,2,3,4,5].map((i)=><g key={i} transform={`translate(${80+i*125} 650)`}><path d="M0 0 Q-15 -180 5 -390" fill="none" stroke={colors.ink} strokeWidth="9" strokeDasharray="500" strokeDashoffset={Math.max(0,500-f*7)}/>{[0,1,2,3].map(j=><ellipse key={j} cx={j%2?25:-18} cy={-140-j*65} rx="33" ry="15" fill={colors.paper} transform={`rotate(${j%2?-25:25} ${j%2?25:-18} ${-140-j*65})`}/>)}</g>)}
      <path d="M20 650H780" stroke={colors.ink} strokeWidth="5"/>
    </svg><Progress index={2}/>
  </AbsoluteFill>;
};

const Print: React.FC = () => {
  const f=useCurrentFrame(); const chars='IDEAS WANT TO TRAVEL'.split('');
  return <AbsoluteFill style={scene}><Grain/><NumberTag>03 / PRINT</NumberTag>
    <div style={{position:'absolute',left:70,top:220,width:1780,borderTop:`4px solid ${colors.ink}`,borderBottom:`4px solid ${colors.ink}`,padding:'35px 0'}}>
      <div style={{display:'flex',justifyContent:'space-between'}}>{chars.map((c,i)=><span key={i} style={{fontFamily:'Georgia,serif',fontWeight:900,fontSize:76,opacity:f>i*2?1:.08,transform:`translateY(${f>i*2?0:-30}px)`}}>{c===' '? '\u00a0':c}</span>)}</div>
    </div>
    <RevealText delay={38} style={{position:'absolute',top:520,left:150,width:700}}><div style={{fontSize:67,fontWeight:900}}>c. 1450</div><p style={{fontSize:29,lineHeight:1.5}}>Movable-type printing in Europe accelerated the copying of texts. Knowledge could be duplicated, challenged, and combined at unprecedented scale.</p></RevealText>
    <RevealText delay={55} style={{position:'absolute',right:140,top:520,width:620,fontSize:48,fontWeight:900,lineHeight:1.15,color:colors.coral}}>THE BREAKTHROUGH<br/>WAS NOT JUST<br/>THE MACHINE.<br/>IT WAS THE NETWORK.</RevealText><Progress index={3}/>
  </AbsoluteFill>;
};

const Industry: React.FC = () => {
  const f=useCurrentFrame();
  return <AbsoluteFill style={{...scene,background:colors.ink,color:colors.paper}}><Grain/><div style={{borderColor:colors.paper}}><NumberTag>04 / INDUSTRY</NumberTag></div>
    <RevealText><div style={{fontSize:130,fontWeight:900,letterSpacing:-7,marginTop:80}}>FIRE → MOTION</div></RevealText>
    <RevealText delay={15}><div style={{fontSize:130,fontWeight:900,letterSpacing:-7,color:colors.gold}}>MOTION → SCALE</div></RevealText>
    <div style={{position:'absolute',bottom:130,left:160,display:'flex',gap:55,alignItems:'center'}}>
      {[150,230,165].map((s,i)=><div key={s} style={{width:s,height:s,border:`18px dotted ${i===1?colors.coral:colors.paper}`,borderRadius:'50%',transform:`rotate(${f*(i%2?-.8:1)}deg)`}}><div style={{width:'35%',height:'35%',background:colors.paper,borderRadius:'50%',margin:'32%'}}/></div>)}
    </div>
    <RevealText delay={35} style={{position:'absolute',right:110,bottom:180,width:650}}><div style={{fontSize:68,fontWeight:900}}>18TH–19TH CENTURIES</div><p style={{fontSize:27,lineHeight:1.5,color:'#bdb7aa'}}>Steam power, factories, and railways multiplied human output—while fossil fuels, dangerous labor, and empire exposed innovation’s cost.</p></RevealText>
  </AbsoluteFill>;
};

const Electricity: React.FC = () => {
  const f=useCurrentFrame(); const pulse=(Math.sin(f/6)+1)/2;
  return <AbsoluteFill style={{...scene,background:colors.sky}}><Grain/><NumberTag>05 / ELECTRICITY</NumberTag>
    <RevealText style={{position:'absolute',left:100,top:250,width:790}}><div style={{fontSize:117,fontWeight:900,letterSpacing:-7,lineHeight:.92}}>THE WORLD<br/>SWITCHED ON.</div><p style={{fontSize:29,lineHeight:1.5}}>Electric grids turned an invisible force into a shared utility. Light, motors, telephones, and computation followed.</p></RevealText>
    <div style={{position:'absolute',right:170,top:170,width:650,height:650,border:`4px solid ${colors.ink}`,borderRadius:'50%',boxShadow:`0 0 ${80+pulse*110}px ${colors.paper}`}}>
      <svg viewBox="0 0 650 650"><path d="M320 70C185 70 115 175 150 295c20 68 75 93 91 173h158c17-80 72-105 93-173C527 175 455 70 320 70Z" fill={colors.paper} stroke={colors.ink} strokeWidth="7"/><path d="M245 468h150v55H245zm20 70h110l-30 45h-50z" fill={colors.ink}/><path d="M245 270l55 45 45-100 55 55-70 198" fill="none" stroke={colors.coral} strokeWidth="15"/></svg>
    </div><Progress index={5}/>
  </AbsoluteFill>;
};

const Digital: React.FC = () => {
  const f=useCurrentFrame(); const p=interpolate(f,[0,120],[0,1],{extrapolateRight:'clamp'});
  const nodes=[[270,260],[620,180],[990,310],[1370,190],[1600,500],[1190,690],[720,650],[300,730]];
  return <AbsoluteFill style={scene}><Grain/><NumberTag>06 / THE NETWORK</NumberTag><TimelineLine progress={p}/>
    <RevealText style={{position:'absolute',top:100,right:85,textAlign:'right'}}><div style={{fontSize:102,fontWeight:900,letterSpacing:-6}}>ONE SPECIES.</div><div style={{fontSize:102,fontWeight:900,letterSpacing:-6,color:colors.coral}}>BILLIONS OF LINKS.</div></RevealText>
    <svg style={{position:'absolute',inset:0,width:'100%',height:'100%'}}>{nodes.map((a,i)=>nodes.slice(i+1).map((b,j)=><line key={`${i}-${j}`} x1={a[0]} y1={a[1]} x2={b[0]} y2={b[1]} stroke={colors.ink} strokeWidth="2" opacity={.07}/>))}{nodes.map((n,i)=><g key={i}><circle cx={n[0]} cy={n[1]} r={20+8*Math.sin(f/8+i)} fill={[colors.coral,colors.mint,colors.gold][i%3]}/><circle cx={n[0]} cy={n[1]} r="5" fill={colors.ink}/></g>)}</svg>
    <RevealText delay={35} style={{position:'absolute',left:140,bottom:120,width:760,fontSize:29,lineHeight:1.5}}>The transistor, integrated circuit, internet, and World Wide Web compressed the distance between idea and audience. <b>Information became infrastructure.</b></RevealText><Progress index={6}/>
  </AbsoluteFill>;
};

const Future: React.FC = () => {
  const f=useCurrentFrame();
  return <AbsoluteFill style={{...scene,background:colors.coral}}><Grain/><NumberTag>07 / NEXT</NumberTag>
    <RevealText><div style={{fontSize:150,fontWeight:900,letterSpacing:-9,lineHeight:.84,marginTop:110}}>WHAT WILL<br/>WE BUILD<br/>TOGETHER?</div></RevealText>
    <div style={{position:'absolute',right:80,top:140,width:700,height:700}}><Orbit radius={340} speed={.7} color={colors.paper}/><Orbit radius={250} speed={-1.2} color={colors.gold}/><Orbit radius={155} speed={1.8} color={colors.mint}/><div style={{position:'absolute',left:282,top:282,width:116,height:116,borderRadius:'50%',background:colors.ink,transform:`scale(${1+Math.sin(f/10)*.08})`}}/></div>
    <RevealText delay={30} style={{position:'absolute',left:85,bottom:120,width:720,fontSize:30,lineHeight:1.5}}>Innovation is never neutral. The next chapter—AI, biotechnology, clean energy—asks not only <b>“Can we?”</b> but <b>“Who benefits?”</b></RevealText>
  </AbsoluteFill>;
};

const Credits: React.FC = () => <AbsoluteFill style={{...scene,background:colors.ink,color:colors.paper,alignItems:'center',justifyContent:'center',textAlign:'center'}}><Grain/><RevealText><div style={{fontSize:48,fontWeight:900}}>EVERY TOOL IS A CHOICE.</div><div style={{fontSize:100,fontWeight:900,color:colors.mint,margin:'20px 0'}}>CHOOSE WISELY.</div><p style={{fontSize:22,color:'#aaa49a'}}>A short film about the long arc of human ingenuity.</p></RevealText></AbsoluteFill>;

const scenes=[Intro,Stone,Farm,Print,Industry,Electricity,Digital,Future,Credits].map((C)=>({C,d:150}));

export const HumanInnovation: React.FC = () => <AbsoluteFill>{scenes.reduce<{items:React.ReactNode[];at:number}>((acc,{C,d},i)=>{acc.items.push(<Sequence key={i} from={acc.at} durationInFrames={d}><C/></Sequence>);acc.at+=d;return acc},{items:[],at:0}).items}</AbsoluteFill>;
