import React from "react";
import { Box, Button, Kbd, Text } from "@chakra-ui/react";

interface AnswerOptionsProps {
  handleReview: (rating: "mastered" | "learned" | "almost_learned" | "not_learned") => void;
}

const AnswerOptions: React.FC<AnswerOptionsProps> = ({ handleReview }) => {
  const options = [
    { rating: "mastered", label: "Mastered", key: "/" },
    { rating: "learned", label: "Learned", key: "." },
    { rating: "almost_learned", label: "Almost Learned", key: "," },
    { rating: "not_learned", label: "Not Learned", key: "m" },
  ];

  return (
    <Box
      display="flex"
      flexDirection="column"
      alignContent="center"
      justifyContent="center"
      alignItems="stretch"
      marginTop="1em"
    >
      {options.map((option) => (
        <Box key={option.rating} position="relative" marginBottom="1em">
          <Button
            onClick={() => handleReview(option.rating as any)}
            width="100%"
          >
            {option.label}
          </Button>
          <Text hideBelow="md" position="absolute" top="50%" right="-70%" transform="translateY(-50%)">
            or Press <Kbd>{option.key}</Kbd>
          </Text>
        </Box>
      ))}
    </Box>
  );
};

export default AnswerOptions;
