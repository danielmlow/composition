
%% Place the subject number here:
clear all
sub = 01; %IMPORTANT CHANGE FOR EACH SUBJECT.
recogTask = true; %true=perform task after every run in the frmi. 

%% test run: 4 sentences. 
% myTrials=experimentSLF_Speedy(subject_ID, run_number, test_run, recogTask); %change second parameter to change run, but later runs need 1st run to work. 
myTrials=experimentSLF_Speedy(88, 1, true, true); %test run is set to true. 
commandwindow;

%% Experiment: 64 sentences per run (about 6-7 minutes each run) 
%% Run 1
myTrials=experimentSLF_Speedy(sub, 1, false, recogTask);
commandwindow;

%% Run 2
myTrials=experimentSLF_Speedy(sub, 2, false, recogTask);
commandwindow;

%% Run 3
myTrials=experimentSLF_Speedy(sub, 3, false, recogTask);
commandwindow;

%% Run 4
myTrials=experimentSLF_Speedy(sub, 4, false, recogTask);
commandwindow;

%% Run 5
myTrials=experimentSLF_Speedy(sub, 5, false, recogTask);
commandwindow;

%% Run 6
myTrials=experimentSLF_Speedy(sub, 6, false, recogTask);
commandwindow;
