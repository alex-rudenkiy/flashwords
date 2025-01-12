import React from "react";
import { Text } from "@chakra-ui/react";
import FancyText from "@carefully-coded/react-text-gradient";

interface WordWidgetProps {
  widgetType: number;
  data: {
    foreign_word: string;
    native_word: string;
    description: string;
  };
  isHiddenWord: boolean;
  setIsHiddenWord: (hidden: boolean) => void;
}

const WordWidget: React.FC<WordWidgetProps> = ({
  widgetType,
  data,
  isHiddenWord,
  setIsHiddenWord,
}) => {
  switch (widgetType) {
    case 0:
      return (
        <>
          <Text fontSize="xl">{data.foreign_word}</Text>
          <FancyText
            gradient={{ from: "#17acff", to: "#ff68f0", type: "linear" }}
            animate
            animateDuration={1000}
          >
            <Text
              style={{
                backgroundColor: isHiddenWord ? "black" : "unset",
                cursor: isHiddenWord ? "pointer" : "unset",
              }}
              onClick={() => setIsHiddenWord(false)}
            >
              {data.native_word}
            </Text>
          </FancyText>
          <Text style={{
                visibility: isHiddenWord ? "hidden" : "visible",
              }}
          >{data.description}</Text>
        </>
      );

    case 1:
      return (
        <>
          <Text fontSize="xl">{data.native_word}</Text>
          <FancyText
            gradient={{ from: "#ff68f0", to: "#17acff", type: "linear" }}
            animate
            animateDuration={1000}
          >
            <Text
              style={{
                backgroundColor: isHiddenWord ? "black" : "unset",
                cursor: isHiddenWord ? "pointer" : "unset",
              }}
              onClick={() => setIsHiddenWord(false)}
            >
              {data.foreign_word}
            </Text>
          </FancyText>
        </>
      );

    case 2:
      return (
        <>
          <Text>{data.description}</Text>
          <FancyText
            gradient={{ from: "#17acff", to: "#ff68f0", type: "linear" }}
            animate
            animateDuration={1000}
          >
            <Text
              style={{
                backgroundColor: isHiddenWord ? "black" : "unset",
                cursor: isHiddenWord ? "pointer" : "unset",
              }}
              onClick={() => setIsHiddenWord(false)}
            >
              {data.foreign_word} - {data.native_word}
            </Text>
          </FancyText>
        </>
      );

    default:
      return <Text>Invalid widget type</Text>;
  }
};

export default WordWidget;
