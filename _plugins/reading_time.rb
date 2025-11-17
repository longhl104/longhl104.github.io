# Calculates reading time for a page
# Average reading speed: 200 words per minute

module Jekyll
  module ReadingTimeFilter
    def reading_time(input)
      words_per_minute = 200
      
      # Strip HTML tags and count words
      words = input.gsub(/<\/?[^>]*>/, "").split.size
      
      # Calculate reading time in minutes, minimum 1 minute
      minutes = (words.to_f / words_per_minute).ceil
      minutes = 1 if minutes < 1
      
      if minutes == 1
        "#{minutes} min read"
      else
        "#{minutes} mins read"
      end
    end
  end
end

Liquid::Template.register_filter(Jekyll::ReadingTimeFilter)
