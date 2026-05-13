files = ["0", "p90", "p50", "p10", "n90", "n50", "n10"];

for ii = 1:numel(files)
    data.results = readdictionary("CSTR_validation/"+files(ii)+"/run_1.json"); % results = (keys: string)
    data.t = cell2mat(data.results{"t"}); % curly brackets: indexes into (1x1) cell array (which was necessary for a happy dictionary definition), then we use cell2mat: converts cell array to a matrix
    data.X = cell2mat(data.results{"T"}); % similar logic as above, but for each variable we want to plot

    plot(data.t, data.X, 'linestyle', '-')%, 'color', '#FFFFFF')
    hold on
end
legend(files, "Location", "best")
ylim([375, 500])
hold off

%% Deprecated
% Plots
% if model.plot == true && runs.full == false % checking if we want to see plots, or if too many would need to be made
%     plotvars = ["h", "Q", "T", "Q_C","C_A"; ... % the variables being plotted
%                 "dm", "L / min", "K", "L / min", "mol / L"]; % and their units
%     numPlots = numel(plotvars(1, :));
% 
%     for jj = 1:model.runs
%         data.results = readdictionary(model.folder+"/run_"+jj+".json"); % results = (keys: string), (values: 1x1 cell array: 1024x1 cell)
%         data.fault = char(cell2mat(data.results{"fault"})'); % moulding the cell array back into a well-behaved string telling us the fault ID
% 
%         data.window = 250; % number of data points considered for the moving window average
% 
%         figure(jj)
%         tile = tiledlayout(numPlots,1);
%         % Plot title: Fault ID + magnitude + units
%         if ~ismember(data.fault, ["F3", "F13"]) % again, we discrim- I mean treat these differently since they don't have a disturbance magnitude
%             tile.Title.String = data.fault + " | Magnitude = " + string(data.results("magnitude") + " " + D.details(D.details(:,1) == data.fault, 3));
%             tile.Title.Interpreter = 'none';
%         else
%             tile.Title.String = data.fault;
%             tile.Title.Interpreter = 'none';
%         end
% 
%         for ii = 1:numPlots
%             nexttile
%             data.t = cell2mat(data.results{"t"}); % curly brackets: indexes into (1x1) cell array (which was necessary for a happy dictionary definition), then we use cell2mat: converts cell array to a matrix
%             data.X = cell2mat(data.results{plotvars(1, ii)}); % similar logic as above, but for each variable we want to plot
% 
%             % Plotting the data... finally
%             plot(data.t, data.X, 'linestyle', '-', 'color', '#AAAAAA') 
%             hold on
% 
%             % Indicating where the disturbance happened
%             if max(data.X) == min(data.X)
%                 plot(data.results{"fault time"} .* [1, 1], [data.X .* 0.75, data.X .* 1.25], 'linestyle', '-', 'color', '#FF0000')
%             else
%                 plot(data.results{"fault time"} .* [1, 1], [max(data.X), min(data.X)], 'linestyle', '-', 'color', '#FF0000')
%             end
% 
%             % Showing what the moving window mean looks like
%             plot(data.t, movmean(data.X, data.window), 'linestyle', '--', 'color', '#000FFF')
% 
%             % Making the formatting nice... hopefully
%             xlim([0, model.d_time])
%             ylabel( plotvars(1, ii) + " (" + plotvars(2, ii) + ")" )
%             xticks({})
%             hold off
% 
%         end
%         % Only the bottom plot needs an x axis with a label and ticks
%         % (there's a joke in there somewhere...)
%         xticks(linspace(0, model.d_time, (model.points-1)/9*5/60))
%         xtickformat('%.1f')
%         tile.Padding = 'compact';
%         tile.TileSpacing = 'compact';
% 
%     end
% end