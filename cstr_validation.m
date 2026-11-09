files = ["0", "p90", "p50", "p10", "n90", "n50", "n10"];

for ii = 1:numel(files)
    data.results = readdictionary("CSTR_validation/"+files(ii)+"/run_1.json"); % results = (keys: string)
    data.t = cell2mat(data.results{"t"}); % curly brackets: indexes into (1x1) cell array (which was necessary for a happy dictionary definition), then we use cell2mat: converts cell array to a matrix
    data.X = cell2mat(data.results{"T"}); % similar logic as above, but for each variable we want to plot

    plot(data.t, data.X, 'linestyle', '-')%, 'color', '#FFFFFF')
    hold on
end
legend(["Nominal Value", "+ 90 %", "+ 50 %", "+ 10 %", "- 90 %", "- 50 %", "- 10 %"], "Location", "best")
xlabel("Time (min)")
ylabel("Reactor temperature (K)")
ylim([375, 500])
hold off
