function myTrials=practice(sub, run, test, recogTask)
% To run, type in the command line:

% myTrials=practice(01, 1, false, true) 

% sub = subject ID (01,02,etc)
% run = run number (there are 6 runs so: 1,2,3,4,5, or 6)  NOT 01, 02. 
% test = set to false to run actual experiment, true to run a couple of
% trials. 
% recogTask = true to run recognition task after every run (that lasts less than a minute). 

% After the sentence "Please wait while we prepare the experiment...", press 5 to simulate the fmri trigger
% and wait 8 seconds.

clc;
Screen('Preference', 'SkipSyncTests', 0 );

%% Parameters
text_size = 50; %maybe Adjust using run4.m.
centerRed = 1; %if the red letter in the words is not center make this larger to go to the left (3,4,5) and smaller (1,0,-1,-2) to go to the right. 
% Check below line 85 for yCent
trialDuration=6; 
pauseLastRun = 4; 
word_rate = .15;
runs = 6;
singleWord=true;
KbReleaseWait(-1);

%% Data
if run==1
    %% Load practice
    load practice.mat %practice
    if test
        practice = practice(1:12);
    end
    numTrials=length(practice); %numb of trials per run
    %% Create structure with sentences and unique ids for each sentence
    for k=1:numTrials
        myTrialsAll(k).sentence=practice(k);
        myTrialsAll(k).id=k;
    end
    %% Unique randomization for each participant    TODO: take 1 from each category.
    myTrials = Shuffle(myTrialsAll);
end
%% Open Window
[win,screenRect]=Screen('OpenWindow',0,[127 127 127]);%[0 0 640 480]); %Open up a PTB window
% [win,screenRect]=Screen('OpenWindow',0,[127 127 127],[0 0 320 240]); %Open up a PTB window
% whichScreen = max(Screen('Screens'));
% [ win, windowRect ] = Screen('OpenWindow', whichScreen, [127 127 127]);
HideCursor()                                          
xCent=round(screenRect(3)/2);%-round(screenRect(3)/10);
yCent=round(screenRect(4)/2);
% yCent=320;
%% Instructions
Screen('TextFont',win, 'Arial');
Screen('TextSize',win, text_size);
DrawFormattedText(win,'Leggi attentamente ogni frase. Ti verrà chiesto di queste frasi dopo ogni sezione. \n ------------------------------------ \n \n Attendi mentre prepariamo l`esperimento. \n \n Le frasi appariranno dopo il segno rosso +.','center','center', [], 50);
Screen(win,'flip')
% pause(5) %When commented, it will show instructions until first trigger
% Screen(win,'flip')

%% Run experiment on second trigger
% First trugger:
while true %waiting for the trigger 5%
    [keyPressed keyTime keyCode]=KbCheck;
    if keyPressed
        keyName=KbName(find(keyCode));
        if strcmp(keyName,'5%')
            %             pause(trialDuration) % TODO: see if we need these secons after the official dummy scans.
%             DrawFormattedText(win,'+',xCent,yCent,  [255,0,0], 50);
%             Screen(win,'flip')
% %             pause(2)
%             Screen(win,'flip')
%             pause(2)
%             fakeScannerPulse=GetSecs;
            break
        end
    end
end

% Second trigger
while true %waiting for the trigger 5%
    [keyPressed keyTime keyCode]=KbCheck;
    if keyPressed
        keyName=KbName(find(keyCode));
        if strcmp(keyName,'5%')
            %             pause(trialDuration) % TODO: see if we need these secons after the official dummy scans.
            DrawFormattedText(win,'+',xCent,yCent,  [255,0,0], 50);
            Screen(win,'flip')
            pause(2)
            Screen(win,'flip')
            fakeScannerPulse=GetSecs;
            break
        end
    end
end

%% show practice
for k=1:numTrials
    myTrials(k).wait_start=GetSecs-fakeScannerPulse;
    while GetSecs<fakeScannerPulse+(trialDuration*(k-1))
    end
    myTrials(k).trial_sent_start=GetSecs-fakeScannerPulse;
    sentence=myTrials(k).sentence{1};
    wordInd=[1 strfind(myTrials(k).sentence{1}, ' ')+1];
    word=[];
    cc=0;
    for ii=1:length(wordInd)-1
        cc=cc+1;
        word{cc}=sentence(wordInd(ii):wordInd(ii+1)-2);
    end
    word{end+1}=sentence(wordInd(end):end);
    % show sentences word by word
    if singleWord
        for ii=1:length(word)
            thisWord=word{ii};
            if strfind(thisWord,'.')% fix full stop
                thisWord(strfind(thisWord,'.'))=[];
            end
            tL=length(thisWord);
            if tL== 1
                targL=1;
            elseif tL< 5
                targL=2;
            elseif tL< 9
                targL=3;
            elseif tL<14
                targL=4;
            else
                targL=5;
            end
            Screen('TextFont',win,'Arial')
            Screen('TextSize',win,text_size)
            % fix apostrophes
            if strfind(thisWord,'''')
                targL=targL+length(strfind(thisWord,''''));
                while strcmp(thisWord(targL),'''')
                    targL=targL-1;
                end
            end
            if targL==1
                %         [nx, ny, textboundsTarg] = DrawFormattedText(win, thisWord, xCent-offSet, round(myScreen(4)/2), [0 0 0])
                [nx, ny, textboundsTarg] = DrawFormattedText(win, thisWord(targL), xCent, yCent, [255, 0, 0]);
            else
                [nx, ny, textboundsTarg] = DrawFormattedText(win, thisWord(1:targL-1), xCent+5000, yCent, [127, 127, 127]);
                offSet=textboundsTarg(3)-textboundsTarg(1);
                [nx, ny, textboundsTarg] = DrawFormattedText(win, thisWord, xCent-offSet, yCent, [0 0 0]);
                [nx, ny, textboundsTarg] = DrawFormattedText(win, thisWord(targL), xCent-centerRed, yCent, [255, 0, 0]);
            end
            Screen(win,'Flip')
            WaitSecs(word_rate) %eg, 0.15 (150 miliseconds per word)
        end
    else
        DrawFormattedText(win,myTrials(k).sentence{1},xCent,yCent, [0,0,0], 50);
        Screen(win,'flip')
        pause(3.5)
    end
    DrawFormattedText(win,'+',xCent,yCent,  [255,0,0]);
    Screen(win,'flip')
    myTrials(k).sent_end=GetSecs-fakeScannerPulse;
end
pause(pauseLastRun) %This will last 10 seconds to obtain full HDR of final trial. 
DrawFormattedText(win,'+',xCent,yCent,  [255,0,0]);
Screen(win,'Flip')
WaitSecs(2); 
%% Save 
% save (['output_s' num2str(sub) 'run' num2str(run) '.mat']);
%% Recognition task  %TODO make it optional in the function.
% Here I add "Task" to many variables names
if recogTask
        %% Instructions
    Screen('TextFont',win, 'Arial');
    Screen('TextSize',win, text_size);
    
    DrawFormattedText(win,'Premi il tasto DESTRA se HAI VISTO la frase nell`ultima sezione \n ed il tasto SINISTRA se NON l`HAI VISTO nell`ultima sezione. \n ------------------------------------ \n \n Le frasi appariranno dopo il segno rosso +.','center','center', [0,0,0], 50);
    Screen(win,'flip')
    pause(12); %time showing the instructions
    %% Load and organize data
    load practice_distractors.mat % todo add unseen sentences.
    numTrialsTask = 4; %it's really 8, but 4 unseen/distractos sentences and 4 seen sentences
    trialDurationTask = 6;

    %%
    if run==1
        %% Create structure with sentences and unique ids for each sentence
        for k=1:numTrialsTask
            myTrialsAllTask(k).sentence=practice_distractors(k);
            myTrialsAllTask(k).id=k;
        end
        %% Unique randomization for each participant    TODO: take 1 from each category.
        myTrialsShuffledTask=Shuffle(myTrialsAllTask);% randomise
%         save(['myTrialsShuffledTask' num2str(sub) '.mat'],'myTrialsShuffledTask','numTrialsTask') %saves shuffled practice.
        %% for every run
        myTrialsTaskUnseen0 = myTrialsShuffledTask(1:numTrialsTask); %subselection of 4 practice for this run
        myTrialsTaskUnseen = [myTrialsTaskUnseen0().sentence]'; %extract sentences only
        myTrialsTaskSeen0 = Shuffle([myTrials().sentence]'); %take random sentences out of last run (seen sentences).
        myTrialsTaskSeen  =  myTrialsTaskSeen0(1:numTrialsTask); %Choose 4
        for k=1:numTrialsTask
            myTrialsTask(k).sentence=myTrialsTaskSeen(k);
            myTrialsTask(k).correctResponse=0;
        end
        for k=1:numTrialsTask
            myTrialsTask(k+numTrialsTask).sentence=myTrialsTaskUnseen(k);
            myTrialsTask(k+numTrialsTask).correctResponse=1;
        end
        myTrialsTask = Shuffle(myTrialsTask);
    end
    %% Recognition task
    DrawFormattedText(win,'+',xCent,yCent,  [255,0,0], 50);
    Screen(win,'flip')
    pause(2)
    Screen(win,'flip')
    fakeScannerPulse = GetSecs;
    for k=1:(numTrialsTask*2)
        myTrialsTask(k).wait_start=GetSecs-fakeScannerPulse;
        while GetSecs<fakeScannerPulse+(trialDurationTask*(k-1))
        end
        myTrialsTask(k).trial_sent_start=GetSecs-fakeScannerPulse;
        sentence=myTrialsTask(k).sentence{1};
        wordInd=[1 strfind(myTrialsTask(k).sentence{1}, ' ')+1];
        word=[];
        cc=0;
        for ii=1:length(wordInd)-1
            cc=cc+1;
            word{cc}=sentence(wordInd(ii):wordInd(ii+1)-2);
        end
        word{end+1}=sentence(wordInd(end):end);
        % show sentences word by word
        if singleWord
            for ii=1:length(word)
                thisWord=word{ii};
                if strfind(thisWord,'.')% fix full stop
                    thisWord(strfind(thisWord,'.'))=[];
                end
                tL=length(thisWord);
                if tL== 1
                    targL=1;
                elseif tL< 5
                    targL=2;
                elseif tL< 9
                    targL=3;
                elseif tL<14
                    targL=4;
                else
                    targL=5;
                end
                Screen('TextFont',win,'Arial')
                Screen('TextSize',win,text_size)
                % fix apostrophes
                if strfind(thisWord,'''')
                    targL=targL+length(strfind(thisWord,''''));
                    while strcmp(thisWord(targL),'''')
                        targL=targL-1;
                    end
                end
                if targL==1
                    %         [nx, ny, textboundsTarg] = DrawFormattedText(win, thisWord, xCent-offSet, round(myScreen(4)/2), [0 0 0])
                    [nx, ny, textboundsTarg] = DrawFormattedText(win, thisWord(targL), xCent, yCent, [255, 0, 0]);
                else
                    [nx, ny, textboundsTarg] = DrawFormattedText(win, thisWord(1:targL-1), xCent+5000, yCent, [127, 127, 127]);
                    offSet=textboundsTarg(3)-textboundsTarg(1);
                    [nx, ny, textboundsTarg] = DrawFormattedText(win, thisWord, xCent-offSet, yCent, [0 0 0]);
                    [nx, ny, textboundsTarg] = DrawFormattedText(win, thisWord(targL), xCent-centerRed, yCent, [255, 0, 0]);
                end
                Screen(win,'Flip')
                WaitSecs(word_rate) %eg, 0.15 (150 miliseconds per word)
            end
        else
            DrawFormattedText(win,myTrialsTask(k).sentence{1},xCent,yCent, [0,0,0], 50);
            Screen(win,'flip')
            pause(3.5)
        end
        %  Record response        
        keyPressed = 0;
        DrawFormattedText(win,'+',xCent,yCent,  [255,0,0]);
        Screen(win,'flip')
        start = GetSecs;
        while GetSecs < start+trialDuration% up to 6 seconds
            [keyPressed keyTime keyCode]=KbCheck
            if keyPressed
                keyName=KbName(keyCode);
                myTrialsTask(k).Resp=(keyName(1));
                myRT_text = GetSecs-start;
                myTrialsTask(k).RT=myRT_text;
            end
        end
    end
end
sca

%% Save
% save (['output_s' num2str(sub) 'run' num2str(run) '.mat']);
% if recogTask
%     save(['myTrialsTask' num2str(sub) 'run' num2str(run) '.mat'],'myTrialsTask') %saves responses from task.
% end

end



