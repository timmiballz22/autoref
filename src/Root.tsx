import {Composition} from 'remotion';
import {HumanInnovation} from './HumanInnovation';

export const RemotionRoot: React.FC = () => (
  <Composition
    id="HumanInnovation"
    component={HumanInnovation}
    durationInFrames={1350}
    fps={30}
    width={1920}
    height={1080}
  />
);
