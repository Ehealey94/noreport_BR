eeglab

%% setting up a for loop
subject ={'P1_', 'P2', 'P3', 'P4', 'P5', 'P6','P7', 'P9', 'P10', 'P11','P12','P13','P14','P15','P16','P17'};
inpath=('C:\Users\evanl\Documents\Second Data Collection\Preprocessed\All Files Preprocessed');
outpath=('C:\Users\evanl\Documents\Second Data Collection\Plots & Data Analysis\power_values\switches\');
ress_dir = 'C:\Users\evanl\Documents\Second Data Collection\4_RESS_v2\'; % folder in which to save RESS struct

% Set up filepath
cd(inpath)
files=dir('*switches*.set')

% Empty cell for results
results={};

% Empty cell for SNR and channel inspection
inspect_snr = {};
inspect_chans = {};
occipital = {'POz', 'PO3', 'PO4', 'PO7', 'PO8', 'O1', 'Oz', 'O2', 'Iz'};

for i = 1:length(files)
    clear ress
    filename=files(i).name
    cd(inpath)
    EEG=pop_loadset(filename, inpath);
    filename=string(erase(files(i).name, ".set"));
    if contains(filename, '.bdf')
        filename = extractBefore(filename, '.bdf');
    end
    char_f=char(filename)
    results{i,1}=char_f(1:3); %Participant

    %% specify parameters
    % response frequencies (you can also try harmonics)
    if contains(filename, 'slow')
        peakfreq1 =14; % hz
        peakfreq2 =17; % hz
        results{i,3}='slow';
    elseif contains(filename, 'fast')
        peakfreq1 =29; % hz
        peakfreq2 =34; % hz
        results{i,3}='fast';
    end
    % parameters for RESS:
    peakwidt  = .5; % FWHM at peak frequency
    neighfreq = 2;  % distance of neighboring frequencies away from peak frequency, +/- in Hz
    neighwidt = 2;  % FWHM of the neighboring frequencies

    % Change the trial length (tl) in secs here to change the axis for the plots and the tidx variable
    if contains(filename, '_1_')
        tl = 60;
        results{i,2}='1'
    elseif contains(filename, '_6_')
        tl=360;
        results{i,2}='6'
    end
    tlms = (tl*1000)-2;

    %% load in the data
    EEG.data = double(EEG.data);

    % both lines automatically change the place at which you index for the specific hz
    hzindex1 = (peakfreq1*10)+1;
    hzindex2 = (peakfreq2*10)+1;

    %% Start RESS for peak frequency '1'

    % FFT parameters 
    nfft = ceil(EEG.srate/.1 ); % .1 Hz resolution
    tidx = dsearchn(EEG.times',[500 tlms]');

    % extract EEG data
    data  = EEG.data; 
    dataX = mean(abs(fft(data(:,tidx(1):tidx(2)),nfft,2)/diff(tidx) ).^2,3); % fast fourier transform for best electrode
    hz    = linspace(0,EEG.srate,nfft); % resolution of SNR calculations and plots
    ress.hz = hz;

    

    % Changing the chanlocs to a cell array for indexing
    D = struct2cell(EEG.chanlocs);


    %% compute covariance matrix at peak frequency
    fdatAt = filterFGx(data,EEG.srate,peakfreq1,peakwidt); % narrowband filter
    fdatAt = reshape( fdatAt(:,tidx(1):tidx(2),:), EEG.nbchan,[] ); % reshape matrix
    fdatAt1 = fdatAt; % saving matrix 
    fdatAt = bsxfun(@minus,fdatAt,mean(fdatAt,2));
    covAt  = (fdatAt*fdatAt')/diff(tidx);

    % compute covariance matrix for lower neighbor
    fdatLo = filterFGx(data,EEG.srate,peakfreq1+neighfreq,neighwidt);
    fdatLo = reshape( fdatLo(:,tidx(1):tidx(2),:), EEG.nbchan,[] );
    fdatLo = bsxfun(@minus,fdatLo,mean(fdatLo,2));
    covLo  = (fdatLo*fdatLo')/diff(tidx);

    % compute covariance matrix for upper neighbor
    fdatHi = filterFGx(data,EEG.srate,peakfreq1-neighfreq,neighwidt);
    fdatHi = reshape( fdatHi(:,tidx(1):tidx(2),:), EEG.nbchan,[] );
    fdatHi = bsxfun(@minus,fdatHi,mean(fdatHi,2));
    covHi  = (fdatHi*fdatHi')/diff(tidx);

    % perform generalized eigendecomposition. This is the meat & potatos of RESS
    [evecs,evals] = eig(covAt,(covHi+covLo)/2);
    [~,comp2plot] = max(diag(evals)); % find maximum component
    evecs = bsxfun(@rdivide,evecs,sqrt(sum(evecs.^2,1))); % normalize vectors (not really necessary, but OK)

    % extract components and force sign
    % maps = inv(evecs'); % get maps (this is fine for full-rank matrices)
    % for the topoplots
    maps = covAt * evecs / (evecs' * covAt * evecs); % this works either way
    [~,idx] = max(abs(maps(:,comp2plot))); % find biggest component % can be determined to choose best electrode
    maps = maps * sign(maps(idx,comp2plot)); % force to positive sign


    % reconstruct RESS component time series for SNR
     ress_ts1 = zeros(EEG.pnts,size(data,3));
    for ti=1:size(data,3)
       ress_ts1(:,ti) = evecs(:,comp2plot)'*squeeze(data(:,:,ti));
    end 

    % reconstruct RESS component power time series (at peak frequency)
    ress_ts1_power = zeros(length(fdatAt1),size(fdatAt1,3));
    for ti=1:size(fdatAt1,3)
       ress_ts1_power(:,ti) = evecs(:,comp2plot)'*squeeze(fdatAt1(:,:,ti));
    end 

    % compute SNR spectrum
    ressx = mean(abs( fft(ress_ts1(tidx(1):tidx(2),:),nfft,1)/diff(tidx) ).^2,2);
    snrR = zeros(size(hz));
    skipbins =  5; % .5 Hz, hard-coded!
    numbins  = 20+skipbins; %  2 Hz, also hard-coded!
    % loop over freqs to compute SNR
    for hzi=numbins+1:length(hz)-numbins-1
        numer = ressx(hzi);
        denom = mean( ressx([hzi-numbins:hzi-skipbins hzi+skipbins:hzi+numbins]) );
        snrR(hzi) = numer./denom;
    end
    ress.snr1 = snrR;
    ress.fr1_map2plot1 = maps(:,comp2plot);
    ress.fr1_map2plot2 = dataX(:,dsearchn(hz',peakfreq1));


    %% now repeat for the other frequency/condition

    % compute covariance matrix at peak frequency
    fdatAt = filterFGx(data,EEG.srate,peakfreq2,peakwidt);
    fdatAt = reshape( fdatAt(:,tidx(1):tidx(2),:), EEG.nbchan,[] );
    fdatAt2 = fdatAt;
    fdatAt = bsxfun(@minus,fdatAt,mean(fdatAt,2));
    covAt  = (fdatAt*fdatAt')/diff(tidx);


    % compute covariance matrix for lower neighbor
    fdatLo = filterFGx(data,EEG.srate,peakfreq2+neighfreq,neighwidt);
    fdatLo = reshape( fdatLo(:,tidx(1):tidx(2),:), EEG.nbchan,[] );
    fdatLo = bsxfun(@minus,fdatLo,mean(fdatLo,2));
    covLo  = (fdatLo*fdatLo')/diff(tidx);

    % compute covariance matrix for upper neighbor
    fdatHi = filterFGx(data,EEG.srate,peakfreq2-neighfreq,neighwidt);
    fdatHi = reshape( fdatHi(:,tidx(1):tidx(2),:), EEG.nbchan,[] );
    fdatHi = bsxfun(@minus,fdatHi,mean(fdatHi,2));
    covHi  = (fdatHi*fdatHi')/diff(tidx);

    % perform generalized eigendecomposition. This is the meat & potatos of RESS
    [evecs,evals] = eig(covAt,(covHi+covLo)/2);
    [~,comp2plot] = max(diag(evals)); % find maximum component
    evecs = bsxfun(@rdivide,evecs,sqrt(sum(evecs.^2,1))); % normalize vectors (not really necessary, but OK)

    % extract components and force sign
    % maps = inv(evecs'); % get maps (this is fine for full-rank matrices)
    maps = covAt * evecs / (evecs' * covAt * evecs); % this works either way
    [~,idx] = max(abs(maps(:,comp2plot))); % find biggest component
    maps = maps * sign(maps(idx,comp2plot)); % force to positive sign


    % reconstruct RESS component time series
    ress_ts2 = zeros(EEG.pnts,size(data,3));
    for ti=1:size(data,3)
        ress_ts2(:,ti) = evecs(:,comp2plot)'*squeeze(data(:,:,ti));
    end

    % reconstruct RESS component time series
    ress_ts2_power = zeros(length(fdatAt2),size(fdatAt2,3));
    for ti=1:size(fdatAt2,3)
        ress_ts2_power(:,ti) = evecs(:,comp2plot)'*squeeze(fdatAt2(:,:,ti));
    end

    % compute SNR spectrum
    ressx = mean(abs( fft(ress_ts2(tidx(1):tidx(2),:),nfft,1)/diff(tidx) ).^2,2);
    snrR = zeros(size(hz));
    skipbins =  5; % .5 Hz, hard-coded!
    numbins  = 20+skipbins; %  2 Hz, also hard-coded!
    % loop over freqs to compute SNR
    for hzi=numbins+1:length(hz)-numbins-1
        numer = ressx(hzi);
        denom = mean( ressx([hzi-numbins:hzi-skipbins hzi+skipbins:hzi+numbins]) );
        snrR(hzi) = numer./denom;
    end
    ress.snr2 = snrR;
    ress.fr2_map2plot1 = maps(:,comp2plot);
    ress.fr2_map2plot2 = dataX(:,dsearchn(hz',peakfreq2));

    %% Adding in section to save SNR and plots

    % compute SNR and best electrodes
    [~,hz1_index] = min(abs(peakfreq1-hz));
    snr1 = ress.snr1(hz1_index);
    [~,hz2_index] = min(abs(peakfreq2-hz));
    snr2 = ress.snr2(hz2_index);
    ress.SNR = [snr1 snr2];

    % select 10 highest weighted electrodes
    [~,ch1_index] = maxk(ress.fr1_map2plot1, 10);
    for CH = 1:10
        chan1{CH} = EEG.chanlocs(ch1_index(CH)).labels;
    end
    [~,ch2_index] = maxk(ress.fr2_map2plot1, 10);
    for CH = 1:10
        chan2{CH} = EEG.chanlocs(ch2_index(CH)).labels;
    end
    ress.chans = {chan1, chan2};

    % start plotting
    f = figure('visible', 'off');
    xlim = [3 48];
    
    subplot(231)
    plot(ress.hz,ress.snr1,'ro-','linew',1,'markersize',5,'markerface','w')
    set(gca,'xlim',xlim)
    axis square
    xlabel('Frequency (Hz)'), ylabel('SNR')
    
    subplot(232)
    map2plot = ress.fr1_map2plot1;
    topoplot(map2plot./max(map2plot),EEG.chanlocs,'maplimits',[-.7 .7],'numcontour',0,'conv','on','electrodes','off','shading','interp');
    title([ 'RESS for ' num2str(peakfreq1) ' Hz' ])

    subplot(233)
    map2plot = ress.fr1_map2plot2;
    topoplot(map2plot./max(map2plot),EEG.chanlocs,'maplimits',[-.7 .7],'numcontour',0,'conv','on','electrodes','off','shading','interp');
    title([ 'RESS for ' num2str(peakfreq1) ' Hz' ])
    title([ 'Electrode power at ' num2str(peakfreq1) ' Hz' ])
    
    subplot(234)
    plot(ress.hz,ress.snr2,'ro-','linew',1,'markersize',5,'markerface','w')
    set(gca,'xlim',xlim)
    axis square
    xlabel('Frequency (Hz)'), ylabel('SNR')

    subplot(235)
    map2plot = ress.fr2_map2plot1;
    topoplot(map2plot./max(map2plot),EEG.chanlocs,'maplimits',[-.7 .7],'numcontour',0,'conv','on','electrodes','off','shading','interp');
    title([ 'RESS for ' num2str(peakfreq2) ' Hz' ])

    subplot(236)
    map2plot = ress.fr2_map2plot2;
    topoplot(map2plot./max(map2plot),EEG.chanlocs,'maplimits',[-.7 .7],'numcontour',0,'conv','on','electrodes','off','shading','interp');
    title([ 'RESS for ' num2str(peakfreq2) ' Hz' ])
    title([ 'Electrode power at ' num2str(peakfreq2) ' Hz' ])

    saveas(f, ['C:\Users\evanl\Documents\Second Data Collection\Plots & Data Analysis\RESS_mapping_v2\' char(filename) '.png']);
    %% Inspect channels and SNR
    if mean(ress.SNR) < 5
        inspect_snr = [inspect_snr, filename];
    end
    % if there are no occipital channels in topography
    common_chans1 = intersect(ress.chans{1}, occipital);
    common_chans2 = intersect(ress.chans{2}, occipital);
    if size(common_chans1,2) < 1 | size(common_chans2,2) < 1
        inspect_chans = [inspect_chans, filename];
    end

    %% This is what Evan created - hilbert transform of the narrowband filtered;

    rh1 = hilbert(ress_ts1_power);
    rh2 = hilbert(ress_ts2_power);
    % take the absolute values of that squared to get the power
    rh1p = (abs(rh1)).^2;
    rh2p = (abs(rh2)).^2;
    rh1p = normalize(rh1p, 'range', [-1, 1]);
    rh2p = normalize(rh2p, 'range', [-1, 1]);

    ress.rh1p = rh1p;
    ress.rh2p = rh2p;
    %% Now compute moving slope 

    % Deleting first and last two seconds of rh1p and rh2p 
    sr=EEG.srate;
    rh1p=rh1p((1+sr*2):(end-sr*2),1);
    rh2p=rh2p((1+sr*2):(end-sr*2),1);


    dydx = movingslope(rh1p, 5);
    low_peaks = find(dydx(1:end-1)>0 & dydx(2:end) < 0);
    msresspwr1 = length(low_peaks);
    
    dydx = movingslope(rh2p, 5);
    high_peaks = find(dydx(1:end-1)>0 & dydx(2:end) < 0);
    msresspwr2 = length(high_peaks);
    
    dydx = movingslope(rh1p, 5);
    low_troughs = find(dydx(1:end-1)<0 & dydx(2:end) > 0);
    msresspwr3 = length(low_troughs);
    
    dydx = movingslope(rh2p, 5);
    high_troughs = find(dydx(1:end-1)<0 & dydx(2:end) > 0);
    msresspwr4 = length(high_troughs)

    switches=[msresspwr1,msresspwr2, msresspwr3 ,msresspwr4];

    results{i, 4} = msresspwr1;
    results{i, 5} = msresspwr2;
    results{i, 6} = msresspwr3;
    results{i,7} = msresspwr4;
    results{i,8} = mean(switches, "omitnan");

    % Save the ress struct
    save([ress_dir, char(filename), '.mat'], 'ress')

    %% Extracting button presses
    if i < 59
        leftIndex = find([EEG.event(:).type] == 49); % For first 58 files
        rightIndex = find([EEG.event(:).type] == 51); % For first 58 files
    elseif i > 58
         leftIndex = find([EEG.event(:).edftype] == 49) % For last 4 files
         rightIndex = find([EEG.event(:).edftype] == 51); % Last 4 files
    end

    leftButton = [EEG.event(leftIndex).latency] / 512;
    leftButton = leftButton(leftButton >= 2 & leftButton <= (tl-2));
       
    rightButton = [EEG.event(rightIndex).latency] / 512;
    rightButton = rightButton(rightButton >= 2 & rightButton <= (tl-2));
    
    dat.leftButton = leftButton;
    dat.rightButton = rightButton;
    dat.buttonSwitches = sort([leftButton rightButton]);
    dat.numSwitches = length(dat.buttonSwitches);
    dat.domDurations = diff(dat.buttonSwitches);

    results{i,9}=length(dat.buttonSwitches);

end

%Save inspect chans and inspect snr
save('inspect_chans.mat', 'inspect_chans')
save('inspect_snr.mat', 'inspect_snr')

%Making a table
col_names={'Participant', 'Length', 'Freq', 'low_peaks', 'high_peaks', 'low_troughs', 'high_troughs', 'ress_switches_ave', 'button_presses'};
t=cell2table(results, 'VariableNames', col_names);
filenames={files(:).name}
t.filenames=filenames.'

writetable(t, 'Evan_results_all_trials.csv')

%Exclude specific trials using 
excl = [inspect_chans inspect_snr];

for i=1:58
    filenames{i}=extractBefore(filenames{i}, '.bdf');
end
for i=59:length(filenames)
    filenames{i}=extractBefore(filenames{i}, '.set');
end

% excl=extractBefore(excl, '.mat');
incl = ~ismember(filenames, excl);

excluded_results_table=t(incl, :)

writetable(excluded_results_table, 'Evan_excluded_results.csv')
