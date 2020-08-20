function Results=FRET_Properties(FRET,FRET_Uncertainty,ForsterRadius,ParameterSpace,Bootstrap_Iterations)

%%%%%%%%%% INPUTS%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FRET:  Matrix containing FRET Efficiency data.  Each row correspond to
% each trajectory and NaNs are used to fill in non-existing data when
% trajectories are different durations or data points are missing.
% 
% FRET_Uncertainty: Uncertainty on each entry in FRET.
% 
% ForsterRadius: Förster radius of the dye pair in nanometers
% 
% ParameterSpace: 3x2 vector containing (from top to bottom) initial
% guesses, lower bounds, and upper bounds of the search region for the
% (from left to right) film thickness in units of R0, and the acceptor
% concentration in units of R0^-2, where R0 is the Förster radius.
% 
% PerformBootstrap: Set to 1 to indicate that the resampling procedure used
% for estimating uncertainty of the parameter should be run.
% 
% Bootstrap_Iterations: Number of resampling iterations used for estimating
% uncertainty of the parameter. Set to 0 if no bootstrap is necessary.
% 
%%%%%%%%%% OUTPUTS%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The results are output as a structure, Results, with the following
% fields:
% 
% Results.Properties: Estimates of film thickness in units of R0, and the
% acceptor concentration in units of R0^-2, where R0 is the Förster radius.
% 
% Results.abc: Parameters describing the FRET between a donor and a plan of
% randomly placed acceptor for use with FRET_Tracking software.
% 
% Results.Prop_Uncertainty: Uncertainty on Properties
% 
% Results.abc_Uncertainty: Uncertainty on abc
% 
% Results.Prop_Samples: Bootstrap_Iterations x 2 matrix containing
% estimated properties for each bootstrap iteration.
% 
% Results.abc_Samples: Bootstrap_Iterations x 2 matrix containing abc for
% each bootstrap iteration.


r0=ForsterRadius;
%Remove trajectories that don't meet duration criteria
fe=FRET;
feu=FRET_Uncertainty;
%Determine film properties
o=optimoptions(@patternsearch,'AccelerateMesh',true,'UseCompletePoll',true,'UseCompleteSearch',false);
[Results.Properties]=patternsearch(@(x)FEMultiASim(feu,fe,x(1),x(2),r0,10000),ParameterSpace(1,:),[],[],[],[],ParameterSpace(2,:),ParameterSpace(3,:),o);
%Determine abc
[~,~,Results.abc]=FEMultiASim(feu,fe,Props(2),Props(1),r0,10000);
for i=1:Bootstrap_Iterations
    %Randomly reasample data with replacement
    samples=randi(size(fe,1),size(fe,1),1);
    FE=fe(samples,:);
    FEu=feu(samples,:);
    %Determine properties
    [Results.Prop_Samples(i,:)]=patternsearch(@(x)FEMultiASim(FEu,FE,x(1),x(2),r0,1000),Props,[],[],[],[],ParameterSpace(2,:),ParameterSpace(3,:),o);
    [~,~,Results.abc_Samples(i,:)]=FEMultiASim(feu,fe,AConc(2),AConc(1),r0,10000);
end
Results.Prop_Uncertainty=std(Results.Properties);
Results.abc_Uncertainty=std(Results.abc);
    function [loglf,FRET,x]=FEMultiASim(FEu,FE,T,Aconc,r0,ND)
        A=Aconc/r0^2;
        FEu=FEu(~isnan(FE));
        FE=FE(~isnan(FE));
        R=[];FRET=[];
        %R is r_k-r_{k-1} where r_k is the lateral distance from the donor
        %to the kth nearest acceptor
        R(:,1)=sqrt(-log(1-rand(ND,1))./(pi*A));
        for k=2:100
            rk=sum(R(:,1:k-1),2);
            R(:,k)=sqrt(-log(1-rand(ND,1))./(pi*A)+rk.^2)-rk; %randomly sample R for ND donors, given rk
        end
        R=cumsum(R,2); %convert R to distance between donor and first 100 nearest acceptors
        %calculate FRET efficiency of simulated data for a range of z
        %positions
        Nz=25;
        for z=1:Nz+1 
            FRET(:,z)=1./(1+(sum((R.^2+((z-1).*T*r0/Nz).^2).^-3*r0^6,2)).^-1);
        end
        z=0:T*r0/Nz:T*r0; %absolute z-positions
        Favg=mean(FRET); %Ensemble average fret of simulated donor
        %fit Favg vs z to surogate function, Femp 
        Femp=@(x,z)1./(1+((x(1)+x(2).*(z/r0).^2)).^x(3));
        x=fminsearch(@(x)sum((Favg-Femp(x,z)).^2),[1/r0^2 1/(pi*A) 3]);
        %estimate likelihood function using a mixture of FRET efficiency
        %distribution based on uniform z-position distribution, and 
        %uncertainty of the FRET efficiency using numerical integration.
        NC=r0*T;
        c1=1./(2*FEu.*FEu);
        c2=exp(-c1.*FE.^2)./sqrt(2*pi*FEu.*FEu);
        loglf=integral(@Intz,0,T*r0,'arrayvalued',1)/NC;
        loglf=-sum(log(nonzeros(loglf(isfinite(loglf)))),'omitnan');
        [T,Aconc];
        if nargout==3
            plot(z,Favg);hold on;plot(z,Femp(x,z));
        end
        function [Val]=Intz(z)
            fe=1./(1+((x(1)+x(2).*(z/r0).^2)).^x(3));
            Val=c2.*exp(c1.*(2*fe.*FE-fe.^2));
        end
        %uncomment the next 2 lines to show plot of the simulated and
        %input FRET distribution
        %evalin('caller','Danhist(fe,-1:.02:2,0);');hold on;Danhist(FRET+FEu(randi(length(FEu),size(FRET))).*randn(size(FRET)),-1:.02:2,0);hold off;
        %shg;
    end
end
