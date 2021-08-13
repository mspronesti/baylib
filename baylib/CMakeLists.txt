install(
        DIRECTORY ./
        DESTINATION include/baylib
        FILES_MATCHING PATTERN "*.hpp"
                       PATTERN "*.h"
)

