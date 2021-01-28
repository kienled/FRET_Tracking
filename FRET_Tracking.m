function Results=FRET_Tracking2(FRET,FRET_Uncertainty,Thickness,CoefGuess,ExpGuess,Trajectory_Duration_Frames,h_params,...
    Max_Consecutive_Failures,N_Samples,Success_per_Sample,CoefTol,ExpTol)

%%%%%%%%%% INPUTS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FRET:  Matrix containing FRET Efficiency data.  Each row correspond to
% each trajectory and NaNs are used to fill in non-existing data when
% trajectories are different durations or data points are missing.
% 
% FRET_Uncertainty: Uncertainty on each entry in FRET.
% 
% Thickness: Film thickness estimate un units of R0, where R0 is the
% Förster radius. The thickness is estimated by the function
% FRET_Properties.
% 
% CoefGuess: Starting guess for the anomalous diffusion coefficient in
% units of R0^2/frame where R0 is the Förster radius.
% 
% ExpGuess: Starting guess for the anomalous diffusion lag-time
% scaling exponent.
% 
% Trajectory_Duration_Frames: Minimum trajectory duration in units of
% frames.  FRET trajectories that do not meet this cutoff will not be used
% and all trajectories will be truncated after the minimum number of
% frames.
% 
% h_params: Parameters (a, b, and c) describing the FRET efficiency between
% a plane of randomly place acceptor and a donor at a distance, z, from the
% acceptor plane, h(z). These parameters are determined using the function
% FRET_Properties.

%%%%%%%%%% OPTIONAL INPUTS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Optional inputs define parameters for the Metropolis-Hastings (MH) and
% expectaion maximization (EM) algorithms. These may be left out, or input
% as [] to used default settings listed below.
% 
% Max_Consecutive_Failures: (Default of 1e5) maximum number of failed
% proposal z positions without success for a given trajectory.  If this is
% reached, the trajectory is discarded by setting all samples to NaN.
% Identity of discarded trajectories can be determined in the output, Zm.
% Exceeding the maximum number of failures can sometimes be avoided by
% reducing MSD_Coef_Guess, which determines the distribution that each
% proposed z-position is drawn from.
% 
% N_Samples: (Default of 250) Number of MH samples to be collected for each
% trajectory
% 
% Success_per_Sample: (Default of 250) Required number proposed z-position
% accepted before recording a sample.  This reduces correlation between
% samples.
% 
% CoefTol: (Default 1e-4) Maximum absolute change in the anomalous
% diffusion coefficient that will end the optimization, if ExpTol is also
% met.
% 
% ExpTol: (Defualt 1E-4) Maximum absolute change in the lag time scaling
% exponent that will end the optimization, if CoefTol is also met.


%%%%%%%%%% OUTPUTS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The results are output as a structure, Results, with the following
% fields:
% 
% Results.MSD_Parameters: 2x1 vector with final estimate of the anomalous
% diffusion coefficient and exponent respectively.  Note that units of the
% coefficient are R0^2/frame.
% 
% Results.MSD_Parameter_CRLB: 2x1 vector with Cramer-Rao lower-bound
% variance estimates for the MSD parameters.  The estimation is performed
% using the numeric 2nd derivative of the expected log-likelihood function
% in place of a true log likelihood. The square root approximate the
% uncertainty on the parameters, but will always underestimate it.
% Resampling methods may provide estimates that are more realistic.
% 
% Results.Zm: Average of the sampled z-positions for each trajectory.  This
% approximates the expected value of the marginal distribution of the
% z-position for each measurement.
% 
% Results.Zu: Standard deviation of the sampled z-positions for each
% trajectory.  This approximates the standard deviation of the marginal
% distribution of the z-position for each measurement.
% 
% Results.Zs: Final set of sampled z-positions for all trajectories
% vertically concatenated into a 2D matrix. This can be used to determine
% the time-averaged MSD.
% 
% Results.TrajInds: Indices of trajectories included in the analysis (i.e.
% those that met the duration criteria).
% 
% Results.Solver_Iteration_Data: Matrix that lists the MSD parameters and
% expected log likelihood before and after the maximization step of each
% iteration for evaluating solver performance. Row 1 is the value of the
% expected log likelihood, row 2 is the value of the anomalous diffusion
% coefficient and row 3 is the value of the lag-time scaling exponent. The
% relative change in the log likelihood with each iteration can be estimate
% by taking the cumulative sum of the increase in the expected log
% likelihood with each maximization step.





tic;
%% initialize and rename variables
G=CoefGuess;
H=ExpGuess/2; %convert to Hurst exponent
Gn=inf; %initial updated parameters
Hn=inf;
DurationFrames=Trajectory_Duration_Frames;
fit=h_params;
jj=0; %counter for collected samples
Evol=[];
Femp=@(x,z)1./(1+((x(1)+x(2).*(z).^2)).^x(3)); %convert z to FRET
Fempinv=@(x,fe)sqrt(((1./fe-1).^(1/x(3))-x(1))/x(2)); %convert FRET to z
TrajInds=find(sum(~isnan(FRET),2)>=DurationFrames);
FEu=FRET_Uncertainty(sum(~isnan(FRET),2)>=DurationFrames,1:DurationFrames);
FE=FRET(sum(~isnan(FRET),2)>=DurationFrames,1:DurationFrames);
FEu(isnan(FE))=NaN;
%% Handle optional inputs or assign default values
if nargin<8||isempty(Max_Consecutive_Failures)
    Nmax=100000;
else
    Nmax=Max_Consecutive_Failures;
end
if nargin<9||isempty(N_Samples)
    NSamp=250;
else
    NSamp=N_Samples;
end
if nargin<10||isempty(Success_per_Sample)
    NSuccess=250;
else
    NSuccess=Success_per_Sample;
end
if nargin<11||isempty(CoefTol)
    CoefTol=.0001;
end
if nargin<12||isempty(ExpTol)
    ExpTol=.0001;
end

%%%%%%% begin Expectation Maximization %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
while 1
    if (abs(G-Gn)<CoefTol&&abs(2*H-2*Hn)<ExpTol) %exit EM if tolerances are met
        break
    end
    %update parameters for next MH sampling
    Gn=G;
    Hn=H;
    %Covariance function for diffusion process
    cov=Gn*((1:DurationFrames-1).^(2*Hn)+((1:DurationFrames-1)').^(2*Hn)-abs((1:DurationFrames-1)-(1:DurationFrames-1)').^(2*Hn));
    icov=cov^-1;
    %MH sampled z-positions, where each cell containes samples from each trajectory
    Z=cell(size(FE,1),1);
    %% filter to generate initial estimates of z-position starting at FE observation of greatest certainty
    i=0;
    fl=Femp(fit,Thickness); %lower bound for true FRET value
    fh=Femp(fit,0); %upper bound for true FRET value
    %set values of FE outside of [fl,fu] to fl or fh in temperary variable FEx
    FEx=FE;
    FEx=max(FEx,fl);
    FEx=min(FEx,fh);
    %Variance estimate on z-position by propagation of error through h(z)
    P=FEu.^2./(4.*fit(2).*fit(3).^2.*FEx.^4.*(1./FEx-1).^(2-2./fit(3)).*((1./FEx - 1).^(1./fit(3))-fit(1)));
    P(imag(P)~=0)=NaN;
    %determine linear index of FRET entry with the greatest confindence in
    %z-space, mfuL
    [~,mfu]=min(P,[],2);
    mfuL=sub2ind(size(FEu),(1:length(mfu))',mfu);
    %initialize filtered fret efficiency, f, and z-position, zz, and assign
    %out of bound initial f to either fl or fh.
    ff=FE(mfuL);
    ffu=FEu(mfuL);
    ff(ff<=fl)=fl+.0001; %added to prevent imaginary numbers when evaluating Femp
    ff(ff>=fh)=fh-.0001;
    zz=zeros(size(FE));
    f=zz;
    f(mfuL)=ff;
    zz(mfuL)=Fempinv(fit,ff);
    for j=1:size(FE,1) %cycle through trajectories
        f(j,mfu(j))=FE(j,mfu(j));
        for t=mfu(j)-1:-1:1 %starting from mfu-1 cycle back to initial point of trajectory
            if isnan(zz(j,t))
                continue
            end
            zz(j,t)=zz(j,t+1);
            %define z postions and covariance for frames
            %t:mfu(j)-1, handling NaNs
            z=fliplr(abs([zz(j,t:mfu(j))]));
            c=cov(1:length(z)-1,1:length(z)-1);
            c=c(~isnan(z(2:end))',~isnan(z(2:end)));
            z=z(~isnan(z));
            %solve for most likely zposition based on already
            %estimated z-positions
            zz(j,t)=fminbnd(@(zn)(Femp(fit,zn)-FE(j,t)).^2./(2*FEu(j,t).*FEu(j,t))-sum(-([z(2:end-1) zn]-z(1))*c^-1*([z(2:end-1) zn]-z(1))','omitnan'),0,Thickness);
        end
        for t=mfu(j)+1:size(FE,2) %starting from mfu+1 cycle forward to final point of trajectory
            if isnan(zz(j,t))
                continue
            end
            zz(j,t)=zz(j,t-1);
            z=abs([zz(j,mfu(j):t)]);
            %define z postions and covariance for frames
            %mfu(j)+1:t, handling NaNs
            c=cov(1:length(z)-1,1:length(z)-1);
            c=c(~isnan(z(2:end))',~isnan(z(2:end)));
            z=z(~isnan(z));
            %solve for most likely zposition based on already
            %estimated z-positions
            zz(j,t)=fminbnd(@(zn)(Femp(fit,zn)-FE(j,t)).^2./(2*FEu(j,t).*FEu(j,t))-sum(-([z(2:end-1) zn]-z(1))*c^-1*([z(2:end-1) zn]-z(1))','omitnan'),0,Thickness);
        end
        f(j,:)=Femp(fit,zz(j,:)); %estimat of true FRET
    end
    Finit=f;
    %% Peform MH sampling of Z-positions for Trajectories in parallel
    parfor j=1:size(FE,1)
        sampRate=25; %number of successes between each check for equilibration
        Z{j}=zeros(NSamp,size(FE,2),'single');
        n=0; %counter for number of successes
        i=0; %counter for number of consecutive failures
        ii=0;  %counter for number of samples
        st=single(sqrt(2*Gn)); %tuning parameters. proposal, zn(t), samples from N(z(t),st)
        p=0;
        init=1; %start with burn in phase (i.e. equilibration phase)
        P=0;
        %initialize z-positions, z, and FRET error, w, using Finit plus
        %random error, and maintain bounds (fl,fh)
        w=randn(1,DurationFrames).*FEu(j,:)+Finit(j,:);
        w(w<fl)=fl+.0001;
        w(w>fh)=fh-.0001;
        z=Fempinv(fit,w);
        w=w-FE(j,:);
        while ii<=NSamp-1
            %propose new z-position from random point in trajectory
            t=randi(DurationFrames);
            zn=z;
            wn=w;
            if isnan(z(t))
                continue
            end
            zn(t)=z(t)+randn()*st;
            wn(t)=1./(1+(fit(1)+fit(2).*zn(t).*zn(t)).^fit(3))-FE(j,t); %direct evaluation is faster than calling Femp
            Pnoise=(-wn(t).*wn(t)+w(t).*w(t))./(2*FEu(j,t).*FEu(j,t)); %log likelihood of w
            %shift to set mean to 0 at t==0
            nzn=(zn(2:end)-zn(1));
            nz=(z(2:end)-z(1));
            t=t-1;
            %calculat log likelihood ration of proposal zn over z
            if t==0 %first frame
                %log likelihood ratio, dp, must be calculated from entire
                %trajectory since initial position is the mean for all z
                nna1=~isnan(nz);
                dp=Pnoise+(-(nzn(nna1))*icov(nna1,nna1)*nzn(nna1)'+nz(nna1)*icov(nna1,nna1)*(nz(nna1))')/2;
            else
                %log likelihood ratio calulated from non-zero terms only
                dp=Pnoise+sum((nz(t)-nzn(t)).*icov(t,[1:t-1 t+1:end]).*nz([1:t-1 t+1:end]),'omitnan')+(nz(t)^2-nzn(t)^2)*icov(t,t)/2;
            end
            t=t+1;
            i=i+1;
            if i>Nmax %exceeded max number of consecutive failures
                Z{j}=NaN*repmat(z,NSamp,1);
                break
            end
            if rand<=exp(dp) %accept proposal at a rate of exp(dp)
                %update parameters
                i=0;
                n=n+1;
                z=zn;
                w=wn;
                p=p+dp;
                if round(n/sampRate)==n/sampRate %check for equilibration or record sample
                    P(n/sampRate+1)=p;
                    %continue cycling until sufficiently equilibrated (i.e. relative probability, P, is randomly changing)
                    if init==1 && (n/sampRate<5 || abs(mean(diff(P(end-4:end)))/P(end-4))>.05)
                        continue
                    end
                    if init==1 %trajectory is equilibrated
                        %update parameters and run 1 additional set of
                        %successes before recording first sample
                        init=0;
                        sampRate=NSuccess;
                        continue
                    end
                    %record sample
                    ii=ii+1;
                    Z{j}(ii,:)=z;
                end
            end
        end
    end
    toc
    %calculate average and standard deviation of the samples for each position
    Zu=cellfun(@(x)std(x,'omitnan'),Z,'UniformOutput',false);
    Zu=cat(1,Zu{:});
    Zm=cellfun(@(x)mean(x,'omitnan'),Z,'UniformOutput',false);
    Zm=cat(1,Zm{:});
    Zm(Zm==0)=NaN;
    %assemble data and estimates corresponding to samples into matrices
    FEs=double(FE(repmat(1:size(FE,1),size(Z{1},1),1),:));
    FEus=double(FEu(repmat(1:size(FEu,1),size(Z{1},1),1),:));
    Zs=double(cat(1,Z{:})); %z position samples
    Fs=double(Femp(fit,Zs)); %FRET samples
    zi=repmat(Zs(:,1),1,size(Zs,2)-1); %mean for estimating probability of z
    if jj==0 %only on the first iteration
        nna=logical([zeros(size(Zs,1),1) ~isnan(Zs(:,2:end))]); %logical matrix indicating missing data (i.e. NaN). First column is zero because mean is defined as initial z-position for mvnpdf.
        %determine unique NaN configureations
        if all(nna(:,2:end),'all') %Check for missing data (i.e. NaN)
            nnaC=ones(size(nna,1),1);
            unnaC=1;
        else
            %convert binary row vectors to base 1+10/Duration and identify unique
            %row vectors
            nnaC=sum((1+10/size(Zs,2)).^(size(Zs,2)-1:-1:0).*nna,2);
            unnaC=unique(nnaC);
        end
    end
    '%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%' %for more attractive display
    Evol=[Evol [sum(sum(log(normpdf(Fs,FEs,FEus)),2,'omitnan')+LFX([2*Hn G]));G;2*Hn]]; %record parameters and likelihood before maximization step
    [S]=fminsearch(@(x)-sum(LFX(x)),double([2*Hn G])); %Maximum likelihood parameters given sampling (maximization step)
    G=S(2);H=S(1)/2;
    Evol=[Evol [sum(sum(log(normpdf(Fs,FEs,FEus)),2,'omitnan')+LFX([2*H G]));G;2*H]]; %record parameters and likelihood after maximization step
    jj=jj+1;
    [CoefGuess,ExpGuess;G H*2] %display current likelihood and parameter values
end
%estimate CRLB on parameters using numerical 2nd derivative of EM maximization function in place of the true log likelihood
c=arrayfun(@(G)sum(log(mvnpdf(Zs(:,2:end),zi,G*((1:DurationFrames-1).^(S(1))+((1:DurationFrames-1)').^(S(1))-abs((1:DurationFrames-1)-(1:DurationFrames-1)').^(S(1))))),'all','omitnan'),S(2)*(.98:.01:1.02));
MSD_Parameter_CRLB(1)=1./(-sum([-1,16,-30,16,-1].*c)./(12*(.01*S(1)).^2));
c=arrayfun(@(H)sum(log(mvnpdf(Zs(:,2:end),Zs(:,1).*ones(1,14),.5*S(2)*((1:DurationFrames-1).^(H)+((1:DurationFrames-1)').^(H)-abs((1:DurationFrames-1)-(1:DurationFrames-1)').^(H)))),'all','omitnan'),S(1)*(.98:.01:1.02));
MSD_Parameter_CRLB(2)=1./(-sum([-1,16,-30,16,-1].*c)./(12*(.01*S(2)).^2));
Results.MSD_Parameters=fliplr(S);
Results.MSD_Parameter_CRLB=MSD_Parameter_CRLB;
Results.Zm=Zm;
Results.Zu=Zu;
Results.Zs=Zs;
Results.Solver_Iteration_Data=Evol;


    function [lf]=LFX(x)
        lf=zeros(size(Zs,1),1);
        cov1=(x(2)*((1:DurationFrames-1).^(x(1))+((1:DurationFrames-1)').^(x(1))-abs((1:DurationFrames-1)-(1:DurationFrames-1)').^(x(1))));
        %cycle through set of trajectories with identical NaN configuration to reduce calls to mvnpdf
        for i=1:length(unnaC)'
            %define indices or subset of trajectories with identical NaN locations
            inds=nnaC==unnaC(i);
            nnat=nna(find(inds,1),:);
            if all(nnat==0)
                continue
            end
            %calculate log likelihood of trajectories
            lf(inds)=log(mvnpdf(Zs(inds,nnat),zi(inds,nnat(2:end)),cov1(nnat(2:end),nnat(2:end))));
        end
    end
end
