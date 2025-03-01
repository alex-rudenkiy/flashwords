//@ts-nocheck
import React, {useEffect, useState} from "react";
import { useQuery, useMutation } from "react-query";
import { Box, Button, Kbd, Spinner, Text } from "@chakra-ui/react";
import axios from "axios";
import FancyText from "@carefully-coded/react-text-gradient";
import { useKeyPressEvent } from "react-use";
import { getToken } from "../Setting";
import WordWidget from "./WordWidget";
import AnswerOptions from "./AnswerOptions";

// Types
interface WordData {
  id: string;
  foreign_word: string;
  native_word: string;
  description: string;
}

type Rating = "mastered" | "learned" | "almost_learned" | "not_learned";

export default function Review() {
  const { data, refetch, isLoading, error, isInitialLoading, isRefetching } = useQuery<WordData>(
    "nextWord",
    () =>
      axios
        .get("/learning/next/0", {
          headers: { Authorization: `Bearer ${getToken()}` },
        })
        .then((res) => res.data),
    { refetchOnWindowFocus: false }
  );

  const mutation = useMutation(
    (review: { wordId: string; rating: Rating; reviewedAt: string }) =>
      axios.post("/learning/review/0", review, {
        headers: { Authorization: `Bearer ${getToken()}` },
      }),
    { onSuccess: () => refetch() }
  );

  const [isHiddenWord, setIsHiddenWord] = useState(true);

  const handleReview = (rating: Rating) => {
    if (data) {
      mutation.mutate({
        wordId: data.id,
        rating,
        reviewedAt: new Date().toISOString(),
      });
      setIsHiddenWord(true);
    }
  };

  useKeyPressEvent("/", () => handleReview("mastered"));
  useKeyPressEvent(".", () => handleReview("learned"));
  useKeyPressEvent(",", () => handleReview("almost_learned"));
  useKeyPressEvent("m", () => handleReview("not_learned"));

  if (isLoading) return <Spinner />;
  if (error) return <Text>No words available for review</Text>;

  return (
    <Box
      display="flex"
      justifyContent="center"
      alignItems="center"
      height="100vh"
      flexDirection="column"
    >
      {!(isLoading|isInitialLoading||isRefetching) && data ? (
        <WordWidget
          data={data}
          isHiddenWord={isHiddenWord}
          setIsHiddenWord={setIsHiddenWord}
        />
      ) : (
        <Spinner />
      )}

      <AnswerOptions handleReview={handleReview} />
    </Box>
  );
}
