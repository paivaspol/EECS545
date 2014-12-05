%% read in chunked data from sequenced files
%%dir_file_prefix: directory and file prefix combination string
%%numfiles: number of files
function [fv] = read_data(dir_file_prefix, numfiles)
    digit = length(int2str(numfiles));

    for k = 1:numfiles
      str = int2str(k);
      if (length(str) < digit)
          for i = 1:digit-length(str)
              str = strcat('0', str);
          end
      end
      myfilename = strcat(dir_file_prefix, str, '.csv')
      if (k == 1)
        fv = importdata(myfilename);
      else
        fv = vertcat(fv, importdata(myfilename));
      end
    end
end